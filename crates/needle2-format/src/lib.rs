//! Reader for Needle 2's self-contained `.cact` export format.
//! The format definition follows Cactus Compute's public Needle exporter.

use std::{error::Error, fmt};

const TAG: u32 = 0x05E1_2A82;
const HEADER_SIZE: usize = 20;
const RECORD_SIZE: usize = 44;
const DTYPE_FP16: u8 = 1;
const DTYPE_FP32: u8 = 2;
const DTYPE_CQ: u8 = 3;
const DTYPE_RAW: u8 = 4;
const TOKENIZER_HEADER_SIZE: usize = 24;
const TOKENIZER_RECORD_SIZE: usize = 7;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CactError {
    TooShort,
    BadTag(u32),
    BadBounds,
    BadShape,
    BadUtf8,
    MissingTokenizer,
    UnsupportedDtype(u8),
}

impl fmt::Display for CactError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid cact model: {self:?}")
    }
}
impl Error for CactError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelHeader {
    pub tensor_count: u32,
    pub codebook_len: u32,
    pub kv_window: u32,
    pub kv_bits: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorRecord {
    pub dtype: u8,
    pub shape: [u32; 4],
    pub ndim: u8,
    pub offset: u64,
    pub nbytes: u64,
    pub group_size: u32,
    pub bits: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerPiece {
    pub kind: u8,
    pub surface: String,
}

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub pad_id: u32,
    pub eos_id: u32,
    pub bos_id: u32,
    pub unk_id: u32,
    pub add_dummy_prefix: bool,
    pub byte_fallback: bool,
    pub pieces: Vec<TokenizerPiece>,
    scores: Vec<f32>,
    byte_ids: [Option<u32>; 256],
    markers: Vec<(String, u32)>,
}

#[derive(Debug, Clone)]
pub struct CactModel<'a> {
    pub header: ModelHeader,
    pub codebook: &'a [u8],
    pub tensors: Vec<TensorRecord>,
    pub bytes: &'a [u8],
    pub tokenizer: Tokenizer,
}

fn u16_at(b: &[u8], o: usize) -> Result<u16, CactError> {
    b.get(o..o + 2)
        .ok_or(CactError::TooShort)
        .map(|x| u16::from_le_bytes([x[0], x[1]]))
}
fn u32_at(b: &[u8], o: usize) -> Result<u32, CactError> {
    b.get(o..o + 4)
        .ok_or(CactError::TooShort)
        .map(|x| u32::from_le_bytes([x[0], x[1], x[2], x[3]]))
}
fn u64_at(b: &[u8], o: usize) -> Result<u64, CactError> {
    b.get(o..o + 8)
        .ok_or(CactError::TooShort)
        .map(|x| u64::from_le_bytes(x.try_into().unwrap()))
}
fn align64(x: usize) -> usize {
    (x + 63) & !63
}

impl<'a> CactModel<'a> {
    pub fn parse(bytes: &'a [u8]) -> Result<Self, CactError> {
        if bytes.len() < HEADER_SIZE {
            return Err(CactError::TooShort);
        }
        let tag = u32_at(bytes, 0)?;
        if tag != TAG {
            return Err(CactError::BadTag(tag));
        }
        let tensor_count = u32_at(bytes, 4)?;
        let codebook_len = u32_at(bytes, 8)?;
        let kv_window = u32_at(bytes, 12)?;
        let kv_bits = u32_at(bytes, 16)?;
        let cb_end = HEADER_SIZE
            .checked_add(codebook_len as usize * 4)
            .ok_or(CactError::BadBounds)?;
        let dir_end = cb_end
            .checked_add(tensor_count as usize * RECORD_SIZE)
            .ok_or(CactError::BadBounds)?;
        if dir_end > bytes.len() {
            return Err(CactError::BadBounds);
        }
        let mut tensors = Vec::with_capacity(tensor_count as usize);
        for i in 0..tensor_count as usize {
            let o = cb_end + i * RECORD_SIZE;
            let dtype = *bytes.get(o).ok_or(CactError::TooShort)?;
            let ndim = *bytes.get(o + 1).ok_or(CactError::TooShort)?;
            if ndim > 4 {
                return Err(CactError::BadShape);
            }
            let mut shape = [0; 4];
            for (j, value) in shape.iter_mut().enumerate() {
                *value = u32_at(bytes, o + 4 + j * 4)?;
            }
            let offset = u64_at(bytes, o + 20)?;
            let nbytes = u64_at(bytes, o + 28)?;
            let group_size = u32_at(bytes, o + 36)?;
            let bits = u32_at(bytes, o + 40)?;
            if !matches!(dtype, DTYPE_FP16 | DTYPE_FP32 | DTYPE_CQ | DTYPE_RAW) {
                return Err(CactError::UnsupportedDtype(dtype));
            }
            let end = offset.checked_add(nbytes).ok_or(CactError::BadBounds)? as usize;
            if offset as usize > bytes.len() || end > bytes.len() {
                return Err(CactError::BadBounds);
            }
            tensors.push(TensorRecord {
                dtype,
                shape,
                ndim,
                offset,
                nbytes,
                group_size,
                bits,
            });
        }
        let tokenizer_record = tensors
            .iter()
            .find(|t| t.dtype == DTYPE_RAW)
            .ok_or(CactError::MissingTokenizer)?;
        let tokenizer_blob = bytes
            .get(
                tokenizer_record.offset as usize
                    ..(tokenizer_record.offset + tokenizer_record.nbytes) as usize,
            )
            .ok_or(CactError::BadBounds)?;
        let tokenizer = parse_tokenizer(tokenizer_blob)?;
        Ok(Self {
            header: ModelHeader {
                tensor_count,
                codebook_len,
                kv_window,
                kv_bits,
            },
            codebook: &bytes[HEADER_SIZE..cb_end],
            tensors,
            bytes,
            tokenizer,
        })
    }

    pub fn tensor_bytes(&self, index: usize) -> Option<&'a [u8]> {
        let t = self.tensors.get(index)?;
        self.bytes
            .get(t.offset as usize..(t.offset + t.nbytes) as usize)
    }

    /// Decode one exported tensor into f32 values. CQ tensors use the codebook
    /// and Walsh-Hadamard transform stored/defined by the public exporter.
    pub fn tensor_f32(&self, index: usize) -> Result<Vec<f32>, CactError> {
        let record = *self.tensors.get(index).ok_or(CactError::BadBounds)?;
        let blob = self.tensor_bytes(index).ok_or(CactError::BadBounds)?;
        let count = record.shape[..record.ndim as usize]
            .iter()
            .try_fold(1usize, |n, &d| n.checked_mul(d as usize))
            .ok_or(CactError::BadShape)?;
        match record.dtype {
            DTYPE_FP16 => Ok(blob
                .chunks_exact(2)
                .map(|x| f16_to_f32(u16::from_le_bytes([x[0], x[1]])))
                .collect()),
            DTYPE_FP32 => Ok(blob
                .chunks_exact(4)
                .map(|x| f32::from_le_bytes(x.try_into().unwrap()))
                .collect()),
            DTYPE_CQ => dequantize_cq(blob, record, self.codebook, count),
            _ => Err(CactError::UnsupportedDtype(record.dtype)),
        }
    }
}

fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits as u32 & 0x8000) << 16) as u32;
    let exp = (bits >> 10) & 0x1f;
    let frac = (bits & 0x03ff) as u32;
    let value = if exp == 0 {
        if frac == 0 {
            sign
        } else {
            let mut f = frac;
            let mut e = -14i32;
            while f & 0x400 == 0 {
                f <<= 1;
                e -= 1;
            }
            sign | (((e + 127) as u32) << 23) | ((f & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (frac << 13)
    } else {
        sign | (((exp as i32 - 15 + 127) as u32) << 23) | (frac << 13)
    };
    f32::from_bits(value)
}

fn dequantize_cq(
    blob: &[u8],
    record: TensorRecord,
    codebook: &[u8],
    count: usize,
) -> Result<Vec<f32>, CactError> {
    if record.ndim != 2 || record.group_size == 0 || !matches!(record.bits, 2 | 3 | 4 | 5) {
        return Err(CactError::BadShape);
    }
    let out = record.shape[0] as usize;
    let input = record.shape[1] as usize;
    let group = record.group_size as usize;
    let padded = (input + group - 1) / group * group;
    let bits = if record.bits == 5 {
        2
    } else {
        record.bits as usize
    };
    let packed_row = padded * bits / 8;
    let norm_offset = out.checked_mul(packed_row).ok_or(CactError::BadBounds)?;
    if blob.len() < norm_offset + out * (padded / group) * 2 {
        return Err(CactError::BadBounds);
    }
    let cb_f32: Vec<f32> = if record.bits == 5 {
        let c = 1.2240064 / (group as f32).sqrt();
        vec![-c, 0.0, c]
    } else {
        let cb_start = match record.bits {
            2 => 0,
            3 => 4,
            4 => 12,
            _ => unreachable!(),
        };
        let cb_len = 1usize << record.bits;
        (0..cb_len)
            .map(|i| {
                f32::from_le_bytes(
                    codebook[(cb_start + i) * 4..(cb_start + i + 1) * 4]
                        .try_into()
                        .unwrap(),
                )
            })
            .collect()
    };
    let mut result = vec![0.0; count];
    for row in 0..out {
        for group_i in 0..padded / group {
            let norm_pos = norm_offset + (row * (padded / group) + group_i) * 2;
            let norm = f16_to_f32(u16::from_le_bytes([blob[norm_pos], blob[norm_pos + 1]]));
            let mut rotated = vec![0.0f32; group];
            for k in 0..group {
                let bit = row * packed_row * 8 + group_i * group * bits + k * bits;
                let mut idx = 0usize;
                for b in 0..bits {
                    let absolute = bit + b;
                    idx |= (((blob[absolute / 8] >> (absolute % 8)) & 1) as usize) << b;
                }
                if record.bits == 5 {
                    idx = match idx {
                        3 => 0,
                        0 => 1,
                        1 => 2,
                        _ => 1,
                    };
                }
                rotated[k] = cb_f32[idx] * norm;
            }
            for j in 0..group {
                let mut value = 0.0;
                for k in 0..group {
                    value += rotated[k]
                        * if (j & k).count_ones() % 2 == 0 {
                            1.0
                        } else {
                            -1.0
                        };
                }
                let pos = row * input + group_i * group + j;
                if pos < result.len() {
                    result[pos] = value / (group as f32).sqrt();
                }
            }
        }
    }
    Ok(result)
}

fn parse_tokenizer(b: &[u8]) -> Result<Tokenizer, CactError> {
    if b.len() < TOKENIZER_HEADER_SIZE {
        return Err(CactError::TooShort);
    }
    let n = u32_at(b, 0)? as usize;
    let mut out = Tokenizer {
        pad_id: u32_at(b, 4)?,
        eos_id: u32_at(b, 8)?,
        bos_id: u32_at(b, 12)?,
        unk_id: u32_at(b, 16)?,
        add_dummy_prefix: b[20] != 0,
        byte_fallback: b[21] != 0,
        pieces: Vec::with_capacity(n),
        scores: Vec::with_capacity(n),
        byte_ids: [None; 256],
        markers: Vec::new(),
    };
    let mut o = TOKENIZER_HEADER_SIZE;
    for id in 0..n {
        if o.checked_add(TOKENIZER_RECORD_SIZE)
            .ok_or(CactError::BadBounds)?
            > b.len()
        {
            return Err(CactError::BadBounds);
        }
        let score = f32::from_le_bytes(b[o..o + 4].try_into().unwrap());
        let kind = b[o + 4];
        let len = u16_at(b, o + 5)? as usize;
        o += TOKENIZER_RECORD_SIZE;
        let end = o.checked_add(len).ok_or(CactError::BadBounds)?;
        let surface = std::str::from_utf8(b.get(o..end).ok_or(CactError::BadBounds)?)
            .map_err(|_| CactError::BadUtf8)?
            .to_owned();
        if kind == 4 && surface.len() == 6 && surface.starts_with("<0x") && surface.ends_with('>') {
            if let Ok(byte) = u8::from_str_radix(&surface[3..5], 16) {
                out.byte_ids[byte as usize] = Some(id as u32);
            }
        }
        // User-defined markers are marked kind 3 by the exporter.
        if kind == 3 {
            out.markers.push((surface.clone(), id as u32));
        }
        out.scores.push(score);
        out.pieces.push(TokenizerPiece { kind, surface });
        o = end;
    }
    out.markers.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));
    Ok(out)
}

impl Tokenizer {
    fn bpe(&self, segment: &str) -> Vec<u32> {
        let mut symbols: Vec<String> = segment.chars().map(|c| c.to_string()).collect();
        while symbols.len() > 1 {
            let mut best: Option<(f32, usize, String)> = None;
            for i in 0..symbols.len() - 1 {
                let candidate = format!("{}{}", symbols[i], symbols[i + 1]);
                if let Some((_id, score)) =
                    self.pieces.iter().enumerate().find_map(|(id, p)| {
                        (p.surface == candidate).then_some((id, self.scores[id]))
                    })
                {
                    if best.as_ref().map_or(true, |x| score > x.0) {
                        best = Some((score, i, candidate));
                    }
                }
            }
            let Some((_, i, merged)) = best else { break };
            symbols.splice(i..=i + 1, [merged]);
        }
        let mut ids = Vec::new();
        for s in symbols {
            if let Some(id) = self.pieces.iter().position(|p| p.surface == s) {
                ids.push(id as u32);
            } else if self.byte_fallback {
                ids.extend(
                    s.as_bytes()
                        .iter()
                        .filter_map(|b| self.byte_ids[*b as usize]),
                );
            } else {
                ids.push(self.unk_id);
            }
        }
        ids
    }

    pub fn encode(&self, text: &str) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }
        let escaped = text.replace(' ', "▁");
        let escaped = if self.add_dummy_prefix {
            format!("▁{escaped}")
        } else {
            escaped
        };
        let mut ids = Vec::new();
        let mut buf = String::new();
        let mut i = 0;
        while i < escaped.len() {
            let rest = &escaped[i..];
            if let Some((marker, id)) = self.markers.iter().find(|(m, _)| rest.starts_with(m)) {
                ids.extend(self.bpe(&buf));
                buf.clear();
                ids.push(*id);
                i += marker.len();
            } else {
                let c = rest.chars().next().unwrap();
                buf.push(c);
                i += c.len_utf8();
            }
        }
        ids.extend(self.bpe(&buf));
        ids
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let Some(p) = self.pieces.get(id as usize) else {
                continue;
            };
            if p.kind == 4 {
                if let Ok(b) = u8::from_str_radix(&p.surface[3..5], 16) {
                    bytes.push(b);
                }
            } else if p.kind != 1 && p.kind != 2 {
                bytes.extend_from_slice(p.surface.as_bytes());
            }
        }
        String::from_utf8_lossy(&bytes).replace('▁', " ")
    }
}

pub fn aligned_directory_end(header: &ModelHeader) -> usize {
    align64(
        HEADER_SIZE + header.codebook_len as usize * 4 + header.tensor_count as usize * RECORD_SIZE,
    )
}
