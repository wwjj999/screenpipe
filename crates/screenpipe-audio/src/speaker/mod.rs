pub mod embedding;

use std::path::Path;

use anyhow::{anyhow, Result};

pub fn create_session<P: AsRef<Path>>(path: P) -> Result<ort::session::Session> {
    let path = path.as_ref();
    // ort 2.0.0-rc.10 panics from inside its global OnceLock when the ONNX
    // Runtime API can't be initialized (Windows DLL/version mismatch hits
    // `expect("Failed to initialize ORT API")` at lib.rs:188). That panic
    // bubbles up the tokio worker and Sentry — convert it to a normal error
    // so callers fall back gracefully instead of crashing the runtime.
    catch_panic_into_error(|| {
        let session = ort::session::Session::builder()?
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .with_intra_threads(1)?
            .with_inter_threads(1)?
            .commit_from_file(path)?;
        Ok(session)
    })
}

fn catch_panic_into_error<F, T>(f: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(result) => result,
        Err(payload) => {
            let msg = payload
                .downcast_ref::<&'static str>()
                .map(|s| (*s).to_string())
                .or_else(|| payload.downcast_ref::<String>().cloned())
                .unwrap_or_else(|| "unknown panic".to_string());
            Err(anyhow!("ort session init panicked: {}", msg))
        }
    }
}

pub mod embedding_manager;
pub mod models;
mod prepare_segments;
pub use prepare_segments::prepare_segments;
pub mod segment;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn catch_panic_into_error_passes_through_ok() {
        let r: Result<i32> = catch_panic_into_error(|| Ok(7));
        assert_eq!(r.unwrap(), 7);
    }

    #[test]
    fn catch_panic_into_error_passes_through_err() {
        let r: Result<()> = catch_panic_into_error(|| Err(anyhow!("normal failure")));
        let msg = r.unwrap_err().to_string();
        assert!(msg.contains("normal failure"));
        assert!(!msg.contains("panicked"));
    }

    #[test]
    fn catch_panic_into_error_catches_str_panic() {
        let r: Result<()> = catch_panic_into_error(|| panic!("boom"));
        let msg = r.unwrap_err().to_string();
        assert!(msg.contains("ort session init panicked"));
        assert!(msg.contains("boom"));
    }

    #[test]
    fn catch_panic_into_error_catches_string_panic() {
        let r: Result<()> = catch_panic_into_error(|| panic!("formatted: {}", 42));
        let msg = r.unwrap_err().to_string();
        assert!(msg.contains("ort session init panicked"));
        assert!(msg.contains("42"));
    }

    #[test]
    fn catch_panic_into_error_simulates_ort_api_init_panic() {
        // Mirrors the exact panic ort 2.0.0-rc.10 raises at lib.rs:188 when
        // `NonNull::new(api).expect("Failed to initialize ORT API")` triggers.
        let r: Result<()> = catch_panic_into_error(|| panic!("Failed to initialize ORT API"));
        let msg = r.unwrap_err().to_string();
        assert!(msg.contains("ort session init panicked"));
        assert!(msg.contains("Failed to initialize ORT API"));
    }

    #[test]
    fn create_session_returns_err_for_missing_path() {
        // Sanity-check the normal error path still flows through ?-propagation
        // (commit_from_file fails, we return Err, no panic conversion needed).
        let r = create_session("/nonexistent/screenpipe-audio-test-model.onnx");
        assert!(r.is_err());
    }
}
