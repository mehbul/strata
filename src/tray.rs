//! Minimal system-tray control for the background server.
//!
//! The tray owns no engine state; it sends the same commands the CLI does.

use anyhow::Result;
use tray_icon::menu::{Menu, MenuEvent, MenuItem};
use tray_icon::{TrayIconBuilder, TrayIconEvent};
use winit::event_loop::EventLoop;

pub async fn run(listen: String, model: String, ctx: usize) -> Result<()> {
    let event_loop = EventLoop::new()?;
    let start = MenuItem::new("Start", true, None);
    let stop = MenuItem::new("Stop", true, None);
    let status = MenuItem::new("Status", true, None);
    let quit = MenuItem::new("Quit", true, None);
    let menu = Menu::new();
    menu.append_items(&[&start, &stop, &status, &quit])?;

    let icon = tray_icon::Icon::from_rgba(vec![32u8; 32 * 32 * 4], 32, 32)?;
    let _tray = TrayIconBuilder::new()
        .with_menu(Box::new(menu))
        .with_tooltip("Strata")
        .with_icon(icon)
        .build()?;

    let menu_channel = MenuEvent::receiver();
    let _tray_channel = TrayIconEvent::receiver();
    let listen_for_menu = listen.clone();
    let model_for_menu = model.clone();

    event_loop.run(move |_event, _elwt| {
        if let Ok(event) = menu_channel.try_recv() {
            if event.id == start.id() {
                let l = listen_for_menu.clone();
                let m = model_for_menu.clone();
                let c = ctx;
                tokio::spawn(async move { crate::daemon::start(&l, &m, c).await.ok(); });
            } else if event.id == stop.id() {
                crate::daemon::stop(&listen_for_menu).ok();
            } else if event.id == status.id() {
                let l = listen_for_menu.clone();
                tokio::spawn(async move { crate::daemon::status(&l).await.ok(); });
            } else if event.id == quit.id() {
                std::process::exit(0);
            }
        }
    })?;
    Ok(())
}
