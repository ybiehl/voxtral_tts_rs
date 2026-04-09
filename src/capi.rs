//! C-compatible FFI layer for Swift/Xcode integration.
//!
//! Exposes a minimal set of C functions that the macOS app calls via
//! a bridging header. All heavy work is synchronous — callers must
//! dispatch to a background thread.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

// ---------------------------------------------------------------------------
// One-time tracing initialisation
// ---------------------------------------------------------------------------

static TRACING_INIT: OnceLock<()> = OnceLock::new();

fn ensure_tracing() {
    TRACING_INIT.get_or_init(|| {
        tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
            )
            .init();
    });
}

// ---------------------------------------------------------------------------
// Opaque context
// ---------------------------------------------------------------------------

/// Opaque inference context. Swift holds this as `OpaquePointer`.
pub struct VoxtralContext {
    tts: crate::inference::VoxtralTTS,
    /// Last error as a null-terminated C string so `voxtral_last_error` can
    /// return a raw pointer that is immediately safe to convert in Swift.
    last_error: CString,
    /// Frame counter updated atomically during generation so Swift can poll
    /// `voxtral_frames_done` from the main thread while Rust generates on a
    /// background thread.
    frames_done: Arc<AtomicUsize>,
}

impl VoxtralContext {
    fn clear_error(&mut self) {
        self.last_error = CString::new("").unwrap();
    }

    fn set_error(&mut self, msg: &str) {
        self.last_error = CString::new(msg).unwrap_or_else(|_| CString::new("error").unwrap());
    }
}

// ---------------------------------------------------------------------------
// Public C API
// ---------------------------------------------------------------------------

/// Load the Voxtral model from `model_dir`.
///
/// Returns a non-null opaque pointer on success, null on failure.
/// The caller must eventually pass the pointer to `voxtral_free`.
///
/// # Safety
/// `model_dir` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn voxtral_init(model_dir: *const c_char) -> *mut VoxtralContext {
    ensure_tracing();

    if model_dir.is_null() {
        return std::ptr::null_mut();
    }

    let dir_str = match CStr::from_ptr(model_dir).to_str() {
        Ok(s) => s,
        Err(_) => return std::ptr::null_mut(),
    };

    let device = crate::tensor::Device::best_available();
    tracing::info!("voxtral_init: loading from {} on {:?}", dir_str, device);

    match crate::inference::VoxtralTTS::from_dir(std::path::Path::new(dir_str), device) {
        Ok(tts) => Box::into_raw(Box::new(VoxtralContext {
            tts,
            last_error: CString::new("").unwrap(),
            frames_done: Arc::new(AtomicUsize::new(0)),
        })),
        Err(e) => {
            tracing::error!("voxtral_init failed: {}", e);
            std::ptr::null_mut()
        }
    }
}

/// Generate speech from `text` using the named `voice`.
///
/// Writes a 24 kHz 16-bit mono WAV file to `output_path`.
/// Returns 0 on success, -1 on error — call `voxtral_last_error` for details.
///
/// **Blocks until generation is complete (~10–30 s for typical text).**
/// Always call from a background thread.
///
/// # Safety
/// All pointer arguments must be valid null-terminated UTF-8 strings.
/// `ctx` must be a non-null pointer returned by `voxtral_init`.
/// `max_tokens` controls the maximum number of audio tokens generated.
/// Each token group is ~80ms of audio (12.5 frames/s × 37 tokens/frame).
/// Pass 0 to use the default (16384 ≈ 35 seconds).
#[no_mangle]
pub unsafe extern "C" fn voxtral_generate(
    ctx: *mut VoxtralContext,
    text: *const c_char,
    voice: *const c_char,
    output_path: *const c_char,
    max_tokens: c_int,
) -> c_int {
    if ctx.is_null() || text.is_null() || voice.is_null() || output_path.is_null() {
        return -1;
    }

    let ctx = &mut *ctx;

    macro_rules! cstr {
        ($ptr:expr, $field:literal) => {
            match CStr::from_ptr($ptr).to_str() {
                Ok(s) => s,
                Err(_) => {
                    ctx.set_error(concat!($field, " is not valid UTF-8"));
                    return -1;
                }
            }
        };
    }

    let text_str = cstr!(text, "text");
    let voice_str = cstr!(voice, "voice");
    let output_str = cstr!(output_path, "output_path");

    const DEFAULT_MAX_TOKENS: usize = 16384;
    const TEMPERATURE: f32 = 0.7;

    let effective_max_tokens = if max_tokens <= 0 {
        DEFAULT_MAX_TOKENS
    } else {
        max_tokens as usize
    };

    // Reset counter and hand an Arc clone to the generate call so progress
    // can be read from the main thread without any locking.
    ctx.frames_done.store(0, Ordering::Relaxed);
    let frames = Arc::clone(&ctx.frames_done);

    match ctx
        .tts
        .generate(text_str, voice_str, TEMPERATURE, effective_max_tokens, Some(&frames))
    {
        Ok((samples, sample_rate)) => {
            match crate::audio::write_wav_file(output_str, &samples, sample_rate) {
                Ok(()) => {
                    ctx.clear_error();
                    0
                }
                Err(e) => {
                    ctx.set_error(&e.to_string());
                    -1
                }
            }
        }
        Err(e) => {
            ctx.set_error(&e.to_string());
            -1
        }
    }
}

/// Return available preset voice names as a comma-separated C string.
///
/// The returned pointer must be freed with `voxtral_free_string`.
/// Returns null if `ctx` is null.
///
/// # Safety
/// `ctx` must be a non-null pointer returned by `voxtral_init`.
#[no_mangle]
pub unsafe extern "C" fn voxtral_list_voices(ctx: *const VoxtralContext) -> *mut c_char {
    if ctx.is_null() {
        return std::ptr::null_mut();
    }
    let voices = (*ctx).tts.list_voices();
    match CString::new(voices.join(",")) {
        Ok(cs) => cs.into_raw(),
        Err(_) => std::ptr::null_mut(),
    }
}

/// Return the number of audio frames generated so far during an ongoing
/// `voxtral_generate` call.  Safe to call from any thread at any time.
/// Returns 0 if `ctx` is null or generation has not started.
///
/// # Safety
/// `ctx` must be a non-null pointer returned by `voxtral_init`.
#[no_mangle]
pub unsafe extern "C" fn voxtral_frames_done(ctx: *const VoxtralContext) -> c_int {
    if ctx.is_null() {
        return 0;
    }
    (*ctx).frames_done.load(Ordering::Relaxed) as c_int
}

/// Load a custom voice from a `.safetensors` file at runtime.
///
/// `name` is the identifier used in `voxtral_generate` (e.g. "my_voice").
/// `path` is the absolute path to a `.safetensors` file containing a tensor
/// with key `"embedding"` and shape `[N, 3072]`.
///
/// Returns 0 on success, -1 on error. Call `voxtral_last_error` for details.
///
/// # Safety
/// All pointer arguments must be valid null-terminated UTF-8 strings.
/// `ctx` must be a non-null pointer returned by `voxtral_init`.
#[no_mangle]
pub unsafe extern "C" fn voxtral_load_voice(
    ctx: *mut VoxtralContext,
    name: *const c_char,
    path: *const c_char,
) -> c_int {
    if ctx.is_null() || name.is_null() || path.is_null() {
        return -1;
    }
    let ctx = &mut *ctx;

    let name_str = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(_) => { ctx.set_error("name is not valid UTF-8"); return -1; }
    };
    let path_str = match CStr::from_ptr(path).to_str() {
        Ok(s) => s,
        Err(_) => { ctx.set_error("path is not valid UTF-8"); return -1; }
    };

    match ctx.tts.load_custom_voice(name_str, std::path::Path::new(path_str)) {
        Ok(()) => { ctx.clear_error(); 0 }
        Err(e) => { ctx.set_error(&e.to_string()); -1 }
    }
}

/// Remove a custom voice by name.
///
/// Preset voices cannot be removed. Returns 0 if the voice was removed,
/// -1 if it was not found or is a preset.
///
/// # Safety
/// `ctx` must be a non-null pointer returned by `voxtral_init`.
/// `name` must be a valid null-terminated UTF-8 string.
#[no_mangle]
pub unsafe extern "C" fn voxtral_remove_voice(
    ctx: *mut VoxtralContext,
    name: *const c_char,
) -> c_int {
    if ctx.is_null() || name.is_null() {
        return -1;
    }
    let ctx = &mut *ctx;
    let name_str = match CStr::from_ptr(name).to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    if ctx.tts.remove_custom_voice(name_str) { 0 } else { -1 }
}

/// Return the last error message, or an empty string if no error.
///
/// The returned pointer is valid until the next call to `voxtral_generate`
/// on this context. Copy it immediately in Swift with `String(cString:)`.
///
/// # Safety
/// `ctx` must be a non-null pointer returned by `voxtral_init`.
#[no_mangle]
pub unsafe extern "C" fn voxtral_last_error(ctx: *const VoxtralContext) -> *const c_char {
    if ctx.is_null() {
        return b"\0".as_ptr() as *const c_char;
    }
    (*ctx).last_error.as_ptr()
}

/// Free a context returned by `voxtral_init`.
///
/// # Safety
/// `ctx` must be a non-null pointer returned by `voxtral_init` that has not
/// already been freed.
#[no_mangle]
pub unsafe extern "C" fn voxtral_free(ctx: *mut VoxtralContext) {
    if !ctx.is_null() {
        drop(Box::from_raw(ctx));
    }
}

/// Free a string returned by `voxtral_list_voices`.
///
/// # Safety
/// `s` must be a non-null pointer returned by `voxtral_list_voices`.
#[no_mangle]
pub unsafe extern "C" fn voxtral_free_string(s: *mut c_char) {
    if !s.is_null() {
        drop(CString::from_raw(s));
    }
}
