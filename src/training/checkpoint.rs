use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

const MAGIC: &[u8; 4] = b"CK10";
const VERSION: u8 = 1;

pub fn save(path: &Path, tensors: &[(&[usize], &[f32])]) -> Result<(), Box<dyn Error>> {
    let mut w = BufWriter::new(File::create(path)?);
    w.write_all(MAGIC)?;
    w.write_all(&[VERSION])?;
    w.write_all(&(tensors.len() as u32).to_le_bytes())?;
    for (shape, data) in tensors {
        w.write_all(&(shape.len() as u32).to_le_bytes())?;
        for &dim in *shape {
            w.write_all(&(dim as u64).to_le_bytes())?;
        }
        for &v in *data {
            w.write_all(&v.to_le_bytes())?;
        }
    }
    Ok(())
}

pub fn load(path: &Path) -> Result<Vec<(Vec<usize>, Vec<f32>)>, Box<dyn Error>> {
    let mut r = BufReader::new(File::open(path)?);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(format!("not a checkpoint file (bad magic {magic:?})").into());
    }

    let mut ver = [0u8; 1];
    r.read_exact(&mut ver)?;
    if ver[0] != VERSION {
        return Err(format!("unsupported checkpoint version {}", ver[0]).into());
    }

    let count = read_u32(&mut r)? as usize;
    let mut tensors = Vec::with_capacity(count);

    for _ in 0..count {
        let rank = read_u32(&mut r)? as usize;
        let mut shape = Vec::with_capacity(rank);
        for _ in 0..rank {
            shape.push(read_u64(&mut r)? as usize);
        }
        let len: usize = shape.iter().product();
        let mut data = Vec::with_capacity(len);
        for _ in 0..len {
            let mut buf = [0u8; 4];
            r.read_exact(&mut buf)?;
            data.push(f32::from_le_bytes(buf));
        }
        tensors.push((shape, data));
    }

    Ok(tensors)
}

fn read_u32(r: &mut impl Read) -> Result<u32, Box<dyn Error>> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> Result<u64, Box<dyn Error>> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::{load, save};

    #[test]
    fn save_load_roundtrip() {
        let path = std::env::temp_dir().join("cifar10_ck10_roundtrip_test.ck10");
        let w = [1.0f32, 2.0, 3.0, 4.0];
        let b = [0.5f32];
        save(
            &path,
            &[(&[2, 2], w.as_slice()), (&[1], b.as_slice())],
        )
        .unwrap();
        let loaded = load(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].0, vec![2, 2]);
        assert_eq!(loaded[0].1, vec![1.0, 2.0, 3.0, 4.0]);
        assert_eq!(loaded[1].0, vec![1]);
        assert_eq!(loaded[1].1, vec![0.5]);
    }

    #[test]
    fn load_rejects_wrong_magic() {
        let path = std::env::temp_dir().join("cifar10_ck10_bad_magic_test.bin");
        std::fs::write(&path, b"XBAD\x01\x00\x00\x00\x00").unwrap();
        let err = load(&path).unwrap_err();
        assert!(err.to_string().contains("bad magic"), "{err}");
    }

    #[test]
    fn load_rejects_wrong_version() {
        let path = std::env::temp_dir().join("cifar10_ck10_bad_version_test.bin");
        std::fs::write(&path, b"CK10\x02\x00\x00\x00\x00").unwrap();
        let err = load(&path).unwrap_err();
        assert!(err.to_string().contains("version"), "{err}");
    }
}
