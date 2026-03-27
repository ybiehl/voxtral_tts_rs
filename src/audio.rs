//! Audio I/O utilities: WAV read/write, PCM encoding, resampling.

use crate::error::{Result, VoxtralError};
use crate::DEFAULT_SAMPLE_RATE;

/// Load a WAV file and return (samples, sample_rate).
/// Samples are normalized to [-1.0, 1.0] as mono f32.
pub fn load_wav_file(path: &str) -> Result<(Vec<f32>, u32)> {
    let reader = hound::WavReader::open(path)
        .map_err(|e| VoxtralError::Audio(format!("Failed to open {}: {}", path, e)))?;

    let spec = reader.spec();
    let sample_rate = spec.sample_rate;

    let samples: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .into_samples::<f32>()
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(|e| VoxtralError::Audio(format!("Failed to read samples: {}", e)))?,
        hound::SampleFormat::Int => {
            let max_val = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .into_samples::<i32>()
                .collect::<std::result::Result<Vec<_>, _>>()
                .map_err(|e| VoxtralError::Audio(format!("Failed to read samples: {}", e)))?
                .into_iter()
                .map(|s| s as f32 / max_val)
                .collect()
        }
    };

    // Convert stereo to mono if needed
    let samples = if spec.channels == 2 {
        samples
            .chunks(2)
            .map(|pair| (pair[0] + pair.get(1).copied().unwrap_or(0.0)) / 2.0)
            .collect()
    } else {
        samples
    };

    Ok((samples, sample_rate))
}

/// Write samples to a WAV file (24kHz, 16-bit mono).
pub fn write_wav_file(path: &str, samples: &[f32], sample_rate: u32) -> Result<()> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut writer = hound::WavWriter::create(path, spec)
        .map_err(|e| VoxtralError::Audio(format!("Failed to create {}: {}", path, e)))?;

    for &sample in samples {
        let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        writer
            .write_sample(s)
            .map_err(|e| VoxtralError::Audio(format!("Failed to write sample: {}", e)))?;
    }

    writer
        .finalize()
        .map_err(|e| VoxtralError::Audio(format!("Failed to finalize WAV: {}", e)))?;

    Ok(())
}

/// Encode samples to WAV bytes in memory (24kHz, 16-bit mono).
pub fn write_wav_bytes(samples: &[f32], sample_rate: u32) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut buffer = std::io::Cursor::new(Vec::new());
    {
        let mut writer = hound::WavWriter::new(&mut buffer, spec)
            .map_err(|e| VoxtralError::Audio(format!("Failed to create WAV writer: {}", e)))?;

        for &sample in samples {
            let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
            writer
                .write_sample(s)
                .map_err(|e| VoxtralError::Audio(format!("Failed to write sample: {}", e)))?;
        }

        writer
            .finalize()
            .map_err(|e| VoxtralError::Audio(format!("Failed to finalize WAV: {}", e)))?;
    }

    Ok(buffer.into_inner())
}

/// Encode f32 samples to 16-bit signed little-endian PCM bytes.
pub fn encode_pcm_i16(samples: &[f32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(samples.len() * 2);
    for &sample in samples {
        let s = (sample * 32767.0).clamp(-32768.0, 32767.0) as i16;
        bytes.extend_from_slice(&s.to_le_bytes());
    }
    bytes
}

/// Resample audio from source_sr to target_sr using high-quality sinc interpolation.
pub fn resample(samples: &[f32], source_sr: u32, target_sr: u32) -> Result<Vec<f32>> {
    if source_sr == target_sr {
        return Ok(samples.to_vec());
    }

    use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let mut resampler = SincFixedIn::<f32>::new(
        target_sr as f64 / source_sr as f64,
        2.0,
        params,
        samples.len(),
        1,
    )
    .map_err(|e| VoxtralError::Audio(format!("Resampler init failed: {}", e)))?;

    let input = vec![samples.to_vec()];
    let output = resampler
        .process(&input, None)
        .map_err(|e| VoxtralError::Audio(format!("Resampling failed: {}", e)))?;

    Ok(output.into_iter().next().unwrap_or_default())
}

/// Load and preprocess audio for the codec: load, resample to 24kHz, normalize.
pub fn load_and_preprocess(path: &str) -> Result<Vec<f32>> {
    let (samples, sr) = load_wav_file(path)?;

    // Resample to 24kHz if needed
    let samples = resample(&samples, sr, DEFAULT_SAMPLE_RATE)?;

    // Peak normalization to 0.95
    let max_abs = samples.iter().map(|s| s.abs()).fold(0.0f32, f32::max);
    let samples = if max_abs > 0.0 {
        let scale = 0.95 / max_abs;
        samples.iter().map(|&s| s * scale).collect()
    } else {
        samples
    };

    Ok(samples)
}
