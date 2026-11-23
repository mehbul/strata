//! A Modelfile reader, in the shape Ollama's `create -f Modelfile` accepts.
//!
//! Only the language is implemented here: how a parsed file becomes a model on
//! disk belongs to `registry`. Keeping the parser separate makes the edge
//! cases — triple-quoted blocks, comments, quoted parameter values — testable
//! without touching a GPU or a model store.

use anyhow::{bail, Result};
use serde_json::{json, Value};

#[derive(Debug, Clone, Default)]
pub struct Modelfile {
    pub from: String,
    pub parameters: Vec<(String, String)>,
    pub template: Option<String>,
    pub system: Option<String>,
    pub adapter: Option<String>,
    pub licenses: Vec<String>,
    pub messages: Vec<(String, String)>,
    pub renderer: Option<String>,
    pub parser: Option<String>,
    pub capabilities: Vec<String>,
}

/// A triple-quoted block body. The closing `"""` may be on its own line.
fn quoted_block(lines: &[&str], i: &mut usize, first: &str) -> Result<String> {
    let mut out = String::new();
    if let Some(rest) = first.strip_prefix("\"\"\"") {
        if let Some(end) = rest.find("\"\"\"") {
            return Ok(rest[..end].to_string());
        }
        out.push_str(rest);
    }
    while *i < lines.len() {
        let line = lines[*i];
        *i += 1;
        if let Some(end) = line.find("\"\"\"") {
            let tail = &line[..end];
            if !out.is_empty() && !tail.is_empty() {
                out.push('\n');
            }
            out.push_str(tail);
            return Ok(out);
        }
        if !out.is_empty() {
            out.push('\n');
        }
        out.push_str(line);
    }
    bail!("unterminated triple-quoted block")
}

/// Split `KEY value` while tolerating a value that contains spaces.
fn key_value(rest: &str) -> Option<(String, String)> {
    let mut parts = rest.splitn(2, char::is_whitespace);
    let key = parts.next()?.to_string();
    let value = parts.next().unwrap_or("").trim().to_string();
    Some((key, value))
}

fn unquote(s: &str) -> String {
    let s = s.trim();
    if s.len() >= 2 && s.starts_with('"') && s.ends_with('"') {
        s[1..s.len() - 1].to_string()
    } else {
        s.to_string()
    }
}

impl Modelfile {
    pub fn parse(text: &str) -> Result<Self> {
        // PowerShell's `utf8` encoding writes a BOM; treat it as a file marker,
        // not as part of the first directive.
        let text = text.trim_start_matches('\u{feff}');
        let lines: Vec<&str> = text.lines().collect();
        let mut out = Modelfile::default();
        let mut i = 0;
        while i < lines.len() {
            let line = lines[i].trim();
            i += 1;
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (directive, rest) = match key_value(line) {
                Some(pair) => pair,
                None => bail!("cannot read Modelfile line: {line}"),
            };
            match directive.to_ascii_uppercase().as_str() {
                "FROM" => out.from = unquote(&rest),
                "PARAMETER" => {
                    let (k, v) = key_value(&rest)
                        .ok_or_else(|| anyhow::anyhow!("PARAMETER needs a key and value: {rest}"))?;
                    out.parameters.push((k, unquote(&v)));
                }
                "TEMPLATE" => {
                    let v = quoted_block(&lines, &mut i, &rest)?;
                    out.template = Some(v);
                }
                "SYSTEM" => {
                    let v = quoted_block(&lines, &mut i, &rest)?;
                    out.system = Some(v);
                }
                "LICENSE" => {
                    let v = quoted_block(&lines, &mut i, &rest)?;
                    if !v.is_empty() {
                        out.licenses.push(v);
                    }
                }
                "MESSAGE" => {
                    let (role, content) = key_value(&rest)
                        .ok_or_else(|| anyhow::anyhow!("MESSAGE needs a role and content"))?;
                    out.messages.push((role, unquote(&content)));
                }
                "ADAPTER" => out.adapter = Some(unquote(&rest)),
                "RENDERER" => out.renderer = Some(unquote(&rest)),
                "PARSER" => out.parser = Some(unquote(&rest)),
                "CAPABILITIES" => {
                    out.capabilities = rest
                        .split(|c: char| c == ',' || c.is_whitespace())
                        .filter(|s| !s.is_empty())
                        .map(str::to_string)
                        .collect();
                }
                other => bail!("unknown Modelfile directive: {other}"),
            }
        }
        if out.from.is_empty() {
            bail!("Modelfile needs a FROM line");
        }
        Ok(out)
    }

    /// The override object stored beside a derived model and returned by
    /// `/api/show`.
    pub fn to_overrides(&self) -> Value {
        let params: serde_json::Map<String, Value> = self
            .parameters
            .iter()
            .map(|(k, v)| (k.clone(), Value::String(v.clone())))
            .collect();
        let mut out = json!({ "from": self.from });
        if let Some(v) = &self.template {
            out["template"] = json!(v);
        }
        if let Some(v) = &self.system {
            out["system"] = json!(v);
        }
        if let Some(v) = &self.adapter {
            out["adapter"] = json!(v);
        }
        if let Some(v) = &self.renderer {
            out["renderer"] = json!(v);
        }
        if let Some(v) = &self.parser {
            out["parser"] = json!(v);
        }
        if !self.licenses.is_empty() {
            out["license"] = json!(self.licenses);
        }
        if !self.messages.is_empty() {
            out["messages"] = json!(
                self.messages
                    .iter()
                    .map(|(role, content)| json!({ "role": role, "content": content }))
                    .collect::<Vec<_>>()
            );
        }
        if !self.capabilities.is_empty() {
            out["capabilities"] = json!(self.capabilities);
        }
        if !params.is_empty() {
            out["parameters"] = Value::Object(params);
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_full_modelfile_round_trips() {
        let text = r#"
# a comment
FROM qwen35
PARAMETER temperature 0.7
PARAMETER stop "<|end|>"
SYSTEM """
You are brief.
Never lie.
"""
TEMPLATE """{{ .System }}
User: {{ .Prompt }}
Assistant: """
MESSAGE user hello
RENDERER qwen3.5
PARSER qwen3.5
CAPABILITIES tools, vision
"#;
        let m = Modelfile::parse(text).unwrap();
        assert_eq!(m.from, "qwen35");
        assert_eq!(m.parameters.len(), 2);
        assert_eq!(m.parameters[1].1, "<|end|>");
        assert_eq!(m.system.as_deref(), Some("You are brief.\nNever lie."));
        assert!(m.template.as_ref().unwrap().contains("User: {{ .Prompt }}"));
        assert_eq!(m.messages[0].0, "user");
        assert_eq!(m.capabilities, vec!["tools", "vision"]);
        let o = m.to_overrides();
        assert_eq!(o["system"], "You are brief.\nNever lie.");
        assert_eq!(o["parameters"]["temperature"], "0.7");
    }

    #[test]
    fn a_missing_from_is_rejected() {
        assert!(Modelfile::parse("PARAMETER temperature 0.2").is_err());
    }

    #[test]
    fn an_unterminated_block_is_rejected() {
        assert!(Modelfile::parse("FROM m\nSYSTEM \"\"\"never closes").is_err());
    }
}
