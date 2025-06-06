use crate::get_store;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::sync::Arc;
use tauri::Emitter;
use tauri::{Manager, State};
use tauri_plugin_shell::process::{CommandChild, CommandEvent};
use tauri_plugin_shell::ShellExt;
use tauri_plugin_store::Store;
use tokio::sync::Mutex;
use tracing::{debug, error, info};

pub struct SidecarState(pub Arc<tokio::sync::Mutex<Option<SidecarManager>>>);

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct UserCredits {
    #[serde(rename = "user.credits.amount")]
    pub amount: i64,
    #[serde(rename = "user.credits.created_at", default)]
    pub created_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct User {
    #[serde(rename = "user.id", default)]
    pub id: Option<String>,
    #[serde(rename = "user.email", default)]
    pub email: Option<String>,
    #[serde(rename = "user.name", default)]
    pub name: Option<String>,
    #[serde(rename = "user.image", default)]
    pub image: Option<String>,
    #[serde(rename = "user.token", default)]
    pub token: Option<String>,
    #[serde(rename = "user.clerk_id", default)]
    pub clerk_id: Option<String>,
    #[serde(default)]
    pub credits: Option<UserCredits>,
    #[serde(rename = "user.cloud_subscribed", default)]
    pub cloud_subscribed: Option<bool>,
}

impl User {
    pub fn from_store<R: tauri::Runtime>(store: &Store<R>) -> Self {
        Self {
            id: store
                .get("user.id")
                .and_then(|v| v.as_str().map(String::from)),
            email: store
                .get("user.email")
                .and_then(|v| v.as_str().map(String::from)),
            name: store
                .get("user.name")
                .and_then(|v| v.as_str().map(String::from)),
            image: store
                .get("user.image")
                .and_then(|v| v.as_str().map(String::from)),
            token: store
                .get("user.token")
                .and_then(|v| v.as_str().map(String::from)),
            clerk_id: store
                .get("user.clerk_id")
                .and_then(|v| v.as_str().map(String::from)),
            credits: Some(UserCredits {
                amount: store
                    .get("user.credits.amount")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0),
                created_at: store
                    .get("user.credits.created_at")
                    .and_then(|v| v.as_str().map(String::from)),
            }),
            cloud_subscribed: store.get("user.cloud_subscribed").and_then(|v| v.as_bool()),
        }
    }
}

#[tauri::command]
pub async fn stop_screenpipe(
    state: State<'_, SidecarState>,
    _app: tauri::AppHandle,
) -> Result<(), String> {
    debug!("Killing screenpipe");

    #[cfg(target_os = "macos")]
    {
        let mut manager = state.0.lock().await;
        if let Some(manager) = manager.as_mut() {
            if let Some(child) = manager.child.take() {
                if let Err(e) = child.kill() {
                    error!("Failed to kill child process: {}", e);
                }
            }
        }
        match tokio::process::Command::new("pkill")
            .arg("-9")
            .arg("-f")
            .arg("screenpipe")
            .output()
            .await
        {
            Ok(_) => {
                debug!("Successfully killed screenpipe processes");
                Ok(())
            }
            Err(e) => {
                error!("Failed to kill screenpipe processes: {}", e);
                Err(format!("Failed to kill screenpipe processes: {}", e))
            }
        }
    }

    #[cfg(target_os = "windows")]
    {
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
        match tokio::process::Command::new("powershell")
            .arg("-NoProfile")
            .arg("-WindowStyle")
            .arg("hidden")
            .arg("-Command")
            .arg(r#"taskkill.exe /F /T /IM screenpipe.exe"#)
            .creation_flags(CREATE_NO_WINDOW)
            .output()
            .await
        {
            Ok(_) => {
                debug!("Successfully killed screenpipe processes");
                Ok(())
            }
            Err(e) => {
                error!("Failed to kill screenpipe processes: {}", e);
                Err(format!("Failed to kill screenpipe processes: {}", e))
            }
        }
    }

    #[cfg(target_os = "linux")]
    {
        // -15 from gnu man page
        // ref: https://www.gnu.org/software/coreutils/manual/html_node/kill-invocation.html
        let command = format!("pgrep -x screenpipe | xargs -I {{}} kill -15 {{}}",);
        match tokio::process::Command::new("sh")
            .arg("-c")
            .arg(command)
            .output()
            .await
        {
            Ok(_) => {
                debug!("Successfully killed screenpipe processes");
                Ok(())
            }
            Err(e) => {
                error!("Failed to kill screenpipe processes: {}", e);
                Err(format!("Failed to kill screenpipe processes: {}", e))
            }
        }
    }
}

#[tauri::command]
pub async fn spawn_screenpipe(
    state: tauri::State<'_, SidecarState>,
    app: tauri::AppHandle,
    override_args: Option<Vec<String>>,
) -> Result<(), String> {
    let mut manager = state.0.lock().await;
    if manager.is_none() {
        *manager = Some(SidecarManager::new());
    }
    if let Some(manager) = manager.as_mut() {
        manager.spawn(&app, override_args).await
    } else {
        debug!("Sidecar already running");
        Ok(())
    }
}

fn spawn_sidecar(
    app: &tauri::AppHandle,
    override_args: Option<Vec<String>>,
) -> Result<CommandChild, String> {
    let store = get_store(app, None).unwrap();

    let audio_transcription_engine = store
        .get("audioTranscriptionEngine")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or(String::from("default"));

    let ocr_engine = store
        .get("ocrEngine")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or(String::from("default"));

    let monitor_ids = store
        .get("monitorIds")
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();

    let audio_devices = store
        .get("audioDevices")
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();

    let use_pii_removal = store
        .get("usePiiRemoval")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let port = store.get("port").and_then(|v| v.as_u64()).unwrap_or(3030);

    let disable_audio = store
        .get("disableAudio")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let ignored_windows = store
        .get("ignoredWindows")
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();

    let included_windows = store
        .get("includedWindows")
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();

    let deepgram_api_key = store
        .get("deepgramApiKey")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or(String::from("default"));

    let fps = store.get("fps").and_then(|v| v.as_f64()).unwrap_or(0.2);

    let dev_mode = store
        .get("devMode")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let vad_sensitivity = store
        .get("vadSensitivity")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or(String::from("high"));

    let audio_chunk_duration = store
        .get("audioChunkDuration")
        .and_then(|v| v.as_u64())
        .unwrap_or(30);

    let telemetry_enabled = store
        .get("analyticsEnabled")
        .and_then(|v| v.as_bool())
        .unwrap_or(true);

    let use_chinese_mirror = store
        .get("useChineseMirror")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let languages = store
        .get("languages")
        .and_then(|v| v.as_array().cloned())
        .unwrap_or_default();

    let enable_beta = store
        .get("enableBeta")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let enable_frame_cache = store
        .get("enableFrameCache")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let enable_ui_monitoring = store
        .get("enableUiMonitoring")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let data_dir = store
        .get("dataDir")
        .and_then(|v| v.as_str().map(String::from))
        .unwrap_or(String::from("default"));

    let enable_realtime_audio_transcription = store
        .get("enableRealtimeAudioTranscription")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let enable_realtime_vision = store
        .get("enableRealtimeVision")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let _use_all_monitors = store
        .get("useAllMonitors")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    let user = User::from_store(&store);

    println!("user: {:?}", user);
    println!("audio_chunk_duration: {}", audio_chunk_duration);

    let port_str = port.to_string();
    let mut args = vec!["--port", port_str.as_str(), "--enable-pipe-manager"];
    let fps_str = fps.to_string();
    if fps != 0.2 {
        args.push("--fps");
        args.push(fps_str.as_str());
    }

    if audio_transcription_engine != "default" {
        args.push("--audio-transcription-engine");
        let model = if audio_transcription_engine == "screenpipe-cloud" {
            "deepgram"
        } else {
            audio_transcription_engine.as_str()
        };
        args.push(model);
    }

    if ocr_engine != "default" {
        args.push("--ocr-engine");
        let model = ocr_engine.as_str();
        args.push(model);
    }

    if !monitor_ids.is_empty() && monitor_ids[0] != Value::String("default".to_string()) {
        for monitor in &monitor_ids {
            args.push("--monitor-id");
            args.push(monitor.as_str().unwrap());
        }
    }

    if !languages.is_empty() && languages[0] != Value::String("default".to_string()) {
        for language in &languages {
            args.push("--language");
            args.push(language.as_str().unwrap());
        }
    }

    if deepgram_api_key != "default" && deepgram_api_key != "" {
        args.push("--deepgram-api-key");
        let key = deepgram_api_key.as_str();
        args.push(key);
    }

    if !audio_devices.is_empty() && audio_devices[0] != Value::String("default".to_string()) {
        for device in &audio_devices {
            args.push("--audio-device");
            args.push(device.as_str().unwrap());
        }
    }

    if use_pii_removal {
        args.push("--use-pii-removal");
    }

    if disable_audio {
        args.push("--disable-audio");
    }

    if !ignored_windows.is_empty() {
        for window in &ignored_windows {
            args.push("--ignored-windows");
            args.push(window.as_str().unwrap());
        }
    }

    if !included_windows.is_empty() {
        for window in &included_windows {
            args.push("--included-windows");
            args.push(window.as_str().unwrap());
        }
    }
    let current_pid = std::process::id();
    let current_pid_str = current_pid.to_string();
    // Set auto-destruct PID if not in dev mode
    if !dev_mode {
        args.push("--auto-destruct-pid");
        args.push(current_pid_str.as_str());
    }

    if vad_sensitivity != "high" {
        args.push("--vad-sensitivity");
        args.push(vad_sensitivity.as_str());
    }

    let audio_chunk_duration_str = audio_chunk_duration.to_string();
    if audio_chunk_duration != 30 {
        args.push("--audio-chunk-duration");
        args.push(audio_chunk_duration_str.as_str());
    }

    if !telemetry_enabled {
        args.push("--disable-telemetry");
    }

    if enable_beta {
        args.push("--enable-beta");
    }

    if enable_frame_cache {
        args.push("--enable-frame-cache");
    }

    if enable_ui_monitoring {
        args.push("--enable-ui-monitoring");
    }

    if data_dir != "default" && !data_dir.is_empty() {
        args.push("--data-dir");
        args.push(data_dir.as_str());
    }

    if enable_realtime_audio_transcription {
        args.push("--enable-realtime-audio-transcription");
    }

    if enable_realtime_vision {
        args.push("--enable-realtime-vision");
    }

    // if use_all_monitors {
    //     args.push("--use-all-monitors");
    // }

    let disable_vision = store
        .get("disableVision")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);

    if disable_vision {
        args.push("--disable-vision");
    }

    // args.push("--debug");

    let override_args_as_vec = override_args.unwrap_or_default();

    if cfg!(windows) {
        let mut c = app.shell().sidecar("screenpipe").unwrap();
        if use_chinese_mirror {
            c = c.env("HF_ENDPOINT", "https://hf-mirror.com");
        }

        // if a user with credits is provided, add the AI proxy env var api url for deepgram as env var https://ai-proxy.i-f9f.workers.dev/v1/listen
        if user.cloud_subscribed.is_some()
            && (deepgram_api_key == "default" || deepgram_api_key == "")
        {
            c = c.env(
                "DEEPGRAM_API_URL",
                "https://ai-proxy.i-f9f.workers.dev/v1/listen",
            );
            c = c.env("DEEPGRAM_WEBSOCKET_URL", "wss://ai-proxy.i-f9f.workers.dev");
            // Add token if screenpipe-cloud is selected and user has a token
            if user.id.is_some() {
                c = c.env("CUSTOM_DEEPGRAM_API_TOKEN", user.id.as_ref().unwrap());
                args.push("--deepgram-api-key");
                args.push(user.id.as_ref().unwrap());
            }
        }

        c = c.env("SENTRY_RELEASE_NAME_APPEND", "tauri");

        // only supports --enable-realtime-vision for now, avoid adding if already present
        if !args.contains(&"--enable-realtime-vision")
            && override_args_as_vec.contains(&"--enable-realtime-vision".to_string())
        {
            args.extend(override_args_as_vec.iter().map(|s| s.as_str()));
        }
        let c = c.args(&args);

        let (_, child) = c.spawn().map_err(|e| {
            error!("Failed to spawn sidecar: {}", e);
            e.to_string()
        })?;

        info!("Spawned sidecar with args: {:?}", args);

        return Ok(child);
    }

    let mut c = app.shell().sidecar("screenpipe").unwrap();

    if use_chinese_mirror {
        c = c.env("HF_ENDPOINT", "https://hf-mirror.com");
    }

    // if a user with credits is provided, add the AI proxy env var api url for deepgram as env var https://ai-proxy.i-f9f.workers.dev/v1/listen
    if user.cloud_subscribed.is_some() && (deepgram_api_key == "default" || deepgram_api_key == "")
    {
        info!(
            "Adding AI proxy env vars for deepgram: {:?}",
            user.id.as_ref().unwrap()
        );
        c = c.env(
            "DEEPGRAM_API_URL",
            "https://ai-proxy.i-f9f.workers.dev/v1/listen",
        );
        c = c.env("DEEPGRAM_WEBSOCKET_URL", "wss://ai-proxy.i-f9f.workers.dev");
        // Add token if screenpipe-cloud is selected and user has a token
        if user.id.is_some() {
            c = c.env("CUSTOM_DEEPGRAM_API_TOKEN", user.id.as_ref().unwrap());
            args.push("--deepgram-api-key");
            args.push(user.id.as_ref().unwrap());
        }
    }

    c = c.env("SENTRY_RELEASE_NAME_APPEND", "tauri");

    // only supports --enable-realtime-vision for now, avoid adding if already present
    if !args.contains(&"--enable-realtime-vision")
        && override_args_as_vec.contains(&"--enable-realtime-vision".to_string())
    {
        args.extend(override_args_as_vec.iter().map(|s| s.as_str()));
    }
    let c = c.args(&args);

    let result = c.spawn();
    if let Err(e) = result {
        error!("Failed to spawn sidecar: {}", e);
        return Err(e.to_string());
    }

    let (mut rx, child) = result.unwrap();
    let app_handle = app.app_handle().clone();

    tauri::async_runtime::spawn(async move {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stdout(line) => {
                    let log_line = String::from_utf8(line).unwrap();
                    print!("{}", log_line);
                    app_handle.emit("sidecar_log", log_line).unwrap();
                }
                CommandEvent::Stderr(line) => {
                    let log_line = String::from_utf8(line).unwrap();
                    error!("Sidecar stderr: {}", log_line);
                    app_handle
                        .emit("sidecar_log", format!("ERROR: {}", log_line))
                        .unwrap();
                }
                _ => {}
            }
        }
    });

    info!("Spawned sidecar with args: {:?}", args);

    Ok(child)
}
pub struct SidecarManager {
    child: Option<CommandChild>,
    dev_mode: Arc<Mutex<bool>>,
}

impl SidecarManager {
    pub fn new() -> Self {
        Self {
            child: None,
            dev_mode: Arc::new(Mutex::new(false)),
        }
    }

    pub async fn spawn(
        &mut self,
        app: &tauri::AppHandle,
        override_args: Option<Vec<String>>,
    ) -> Result<(), String> {
        info!("Spawning sidecar with override args: {:?}", override_args);
        // Update settings from store
        self.update_settings(app).await?;

        // Spawn the sidecar
        let child = spawn_sidecar(app, override_args)?;
        self.child = Some(child);

        Ok(())
    }

    async fn update_settings(&mut self, app: &tauri::AppHandle) -> Result<(), String> {
        let store = get_store(app, None).unwrap();

        let dev_mode = store
            .get("devMode")
            .and_then(|v| v.as_bool())
            .unwrap_or(false);

        debug!("dev_mode: {}", dev_mode);

        *self.dev_mode.lock().await = dev_mode;

        Ok(())
    }
}
