//! Which layers keep their expert weights in host memory.
//!
//! The runtime has two ways to say this, and they are not equally expressive:
//!
//! * `-ncmoe N` keeps the experts of the **first N layers** on the CPU. One
//!   integer, one cut point, and the choice of *which* layers is not yours.
//! * `-ot <pattern>=<buffer>` overrides the buffer for every tensor matching a
//!   pattern. Tensor names are the vocabulary placement already thinks in
//!   (`blk.<i>.ffn_*_exps.weight`), so any set of layers can be named.
//!
//! `-ncmoe` is a special case of `-ot`. This module represents the general
//! plan and renders whichever form the runtime needs - deliberately preferring
//! `-ncmoe` whenever the plan happens to be a prefix, so the common case
//! produces the exact command line it always has and nothing about a machine
//! that is already tuned moves.

/// The layers whose routed experts live in host memory.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertPlan {
    cpu_layers: Vec<usize>,
    total_layers: usize,
}

impl ExpertPlan {
    /// What `-ncmoe n` means: the first `n` layers.
    pub fn first_n(n: usize, total_layers: usize) -> Self {
        let n = n.min(total_layers);
        Self { cpu_layers: (0..n).collect(), total_layers }
    }

    /// An arbitrary set of layers. Order and duplicates do not matter.
    pub fn from_layers(layers: impl IntoIterator<Item = usize>, total_layers: usize) -> Self {
        let mut cpu_layers: Vec<usize> =
            layers.into_iter().filter(|&l| l < total_layers).collect();
        cpu_layers.sort_unstable();
        cpu_layers.dedup();
        Self { cpu_layers, total_layers }
    }

    pub fn cpu_layers(&self) -> &[usize] {
        &self.cpu_layers
    }

    pub fn len(&self) -> usize {
        self.cpu_layers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.cpu_layers.is_empty()
    }

    /// `Some(n)` when the plan is exactly the first `n` layers, which the
    /// runtime already has a flag for. Returning `Some` is what keeps an
    /// already-tuned machine on the command line it was measured with.
    pub fn as_first_n(&self) -> Option<usize> {
        self.cpu_layers
            .iter()
            .enumerate()
            .all(|(i, &layer)| i == layer)
            .then_some(self.cpu_layers.len())
    }

    /// The `-ot` value placing these layers' expert tensors on the CPU, or
    /// `None` when there is nothing to place.
    ///
    /// Layer numbers are matched with a trailing `\.` so that `blk.1.` cannot
    /// also match `blk.10.`.
    pub fn to_override(&self) -> Option<String> {
        if self.cpu_layers.is_empty() {
            return None;
        }
        let alternatives = self
            .cpu_layers
            .iter()
            .map(|l| l.to_string())
            .collect::<Vec<_>>()
            .join("|");
        Some(format!(r"blk\.({alternatives})\.ffn_.*_exps\.=CPU"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_prefix_plan_is_still_expressed_as_ncmoe() {
        // The configuration measured on the development machine. It must keep
        // resolving to the flag it was measured with, not to an -ot pattern.
        let plan = ExpertPlan::first_n(15, 41);
        assert_eq!(plan.as_first_n(), Some(15));
        assert_eq!(plan.len(), 15);
    }

    #[test]
    fn a_gap_makes_it_a_pattern() {
        let plan = ExpertPlan::from_layers([0, 1, 5], 41);
        assert_eq!(plan.as_first_n(), None);
        assert_eq!(plan.to_override().unwrap(), r"blk\.(0|1|5)\.ffn_.*_exps\.=CPU");
    }

    #[test]
    fn layer_one_does_not_match_layer_ten() {
        let pattern = ExpertPlan::from_layers([1], 41).to_override().unwrap();
        // The trailing escaped dot is what prevents blk.10. from matching.
        assert!(pattern.contains(r"(1)\.ffn_"));
        assert!(!pattern.contains("(1)ffn"));
    }

    #[test]
    fn an_empty_plan_asks_for_nothing() {
        let plan = ExpertPlan::first_n(0, 41);
        assert!(plan.is_empty());
        assert_eq!(plan.to_override(), None);
        assert_eq!(plan.as_first_n(), Some(0));
    }

    #[test]
    fn layers_past_the_end_are_dropped_and_duplicates_collapse() {
        let plan = ExpertPlan::from_layers([3, 3, 99, 1], 41);
        assert_eq!(plan.cpu_layers(), &[1, 3]);
    }

    #[test]
    fn a_prefix_built_the_long_way_is_recognised_as_a_prefix() {
        let plan = ExpertPlan::from_layers([2, 0, 1], 41);
        assert_eq!(plan.as_first_n(), Some(3));
    }
}
