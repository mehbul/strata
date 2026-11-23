//! Windows Service Control Manager integration.
//!
//! `strata service-windows install|uninstall` registers a real service; the
//! service process runs the daemon watchdog, so the engine restarts on failure
//! and starts at boot without a logged-in user.

use anyhow::{bail, Result};
use std::ffi::OsString;
use windows_service::service::{
    ServiceAccess, ServiceErrorControl, ServiceInfo, ServiceStartType, ServiceType,
};
use windows_service::service_manager::{ServiceManager, ServiceManagerAccess};

const NAME: &str = "Strata";
const DISPLAY: &str = "Strata local inference engine";

pub fn install(listen: &str, model: &str, ctx: usize) -> Result<()> {
    let exe = std::env::current_exe()?;
    let mut args = OsString::from("--service-run --listen ");
    args.push(listen);
    args.push(" --model ");
    args.push(model);
    args.push(" --ctx ");
    args.push(ctx.to_string());
    let command = format!("\"{}\" {}", exe.display(), args.to_string_lossy());

    let manager = match ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CREATE_SERVICE) {
        Ok(m) => m,
        Err(e) => bail!("opening Service Manager failed ({e}); run this terminal as Administrator"),
    };
    let info = ServiceInfo {
        name: NAME.into(),
        display_name: DISPLAY.into(),
        service_type: ServiceType::OWN_PROCESS,
        start_type: ServiceStartType::AutoStart,
        error_control: ServiceErrorControl::Normal,
        executable_path: exe,
        launch_arguments: vec![
            "--service-run".into(),
            "--listen".into(),
            listen.into(),
            "--model".into(),
            model.into(),
            "--ctx".into(),
            ctx.to_string().into(),
        ],
        dependencies: vec![],
        account_name: None,
        account_password: None,
    };
    match manager.create_service(&info, ServiceAccess::CHANGE_CONFIG) {
        Ok(service) => {
            service.set_description("Strata OpenAI-compatible local model server")?;
            println!("Installed Windows service {NAME}: {command}");
            Ok(())
        }
        Err(e) => {
            if e.to_string().contains("1073") || e.to_string().to_ascii_lowercase().contains("already exists") {
                bail!("service {NAME} already exists")
            }
            Err(e.into())
        }
    }
}

pub fn uninstall() -> Result<()> {
    let manager = match ServiceManager::local_computer(None::<&str>, ServiceManagerAccess::CREATE_SERVICE) {
        Ok(m) => m,
        Err(e) => bail!("opening Service Manager failed ({e}); run this terminal as Administrator"),
    };
    let service = match manager.open_service(NAME, ServiceAccess::DELETE) {
        Ok(s) => s,
        Err(e) => bail!("opening service {NAME} failed ({e}); run as Administrator if it exists"),
    };
    service.delete()?;
    println!("Removed Windows service {NAME}");
    Ok(())
}

/// Run the service dispatcher. Called from main when `--service-run` is set.
pub fn run_service(listen: String, model: String, ctx: usize) -> Result<()> {
    use windows_service::define_windows_service;
    use windows_service::service_dispatcher;

    define_windows_service!(ffi_service_main, service_main);
    fn service_main(_args: Vec<OsString>) {
        let listen = std::env::var("STRATA_SERVICE_LISTEN").unwrap_or_else(|_| "127.0.0.1:8080".into());
        let model = std::env::var("STRATA_SERVICE_MODEL").unwrap_or_default();
        let ctx: usize = std::env::var("STRATA_SERVICE_CTX")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(8192);
        let rt = tokio::runtime::Runtime::new().expect("tokio runtime");
        rt.block_on(async { crate::daemon::watch(&listen, &model, ctx).await.ok(); });
    }

    std::env::set_var("STRATA_SERVICE_LISTEN", listen);
    std::env::set_var("STRATA_SERVICE_MODEL", model);
    std::env::set_var("STRATA_SERVICE_CTX", ctx.to_string());
    service_dispatcher::start(NAME, ffi_service_main)?;
    Ok(())
}
