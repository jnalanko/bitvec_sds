//! Port of sdsl::select_support_mcl to Rust, using bitvec for the bitvector.
//!
//! - Superblock size: 4096 occurrences
//! - For each superblock: store position of first occurrence (L1 sample)
//!   * If span >= log^4(n): store all positions explicitly ("long" block)
//!   * else: store relative position of every 64th occurrence ("mini" block) and
//!           answer within-block via word scans (constant-ish time)
//!
//! Supports patterns: 1, 0, 10, 01 (pattern length 1 or 2).
//!
//! Convention: select(i) is 1-based: select(1) = position of 1st occurrence.

use bitvec::prelude::*;
use std::marker::PhantomData;
use std::sync::Arc;


// ============================
// select_support_mcl structure
// ============================

#[derive(Clone, Debug)]
enum Block {
    /// Explicit positions for all occurrences in this superblock (size <= 4096).
    Long(Vec<usize>),
    /// Miniblock relative offsets for every 64th occurrence (always 64 entries),
    /// plus count of occurrences in this superblock (needed for last partial block).
    Mini { mini: [u32; 64], cnt: usize },
}

impl Block {
    fn serialize(&self, mut writer: &mut impl std::io::Write) {
        // TODO: all blocks at once instead of one by one, to reduce overhead of bincode calls.
        match self {
            Block::Long(pos) => {
                bincode::serialize_into(&mut writer, &0u32).unwrap(); // marker for long block
                bincode::serialize_into(&mut writer, pos).unwrap();
            }
            Block::Mini { mini, cnt } => {
                bincode::serialize_into(&mut writer, &1u32).unwrap(); // marker for mini block
                writer.write_all(bytemuck::cast_slice(mini)).unwrap();
                bincode::serialize_into(&mut writer, cnt).unwrap();
            }
        }
    }

    fn load(mut reader: &mut impl std::io::Read) -> Self {
        let mut marker_buf = [0u8; 4];
        reader.read_exact(&mut marker_buf).unwrap();
        let marker = u32::from_le_bytes(marker_buf);
        match marker {
            0 => {
                let pos: Vec<usize> = bincode::deserialize_from(&mut reader).unwrap();
                Block::Long(pos)
            }
            1 => {
                let mut mini_buf = [0u8; 64 * 4];
                reader.read_exact(&mut mini_buf).unwrap();
                let mini: [u32; 64] = bytemuck::cast_slice(&mini_buf).try_into().unwrap();

                let mut cnt_buf = [0u8; 8];
                reader.read_exact(&mut cnt_buf).unwrap();
                let cnt = usize::from_le_bytes(cnt_buf);

                Block::Mini { mini, cnt }
            }
            _ => panic!("Invalid block marker"),
        }
    }

}

#[derive(Clone, Debug)]
pub struct SelectSupportMcl<P: SelectTrait = Sel1> {
    bv: Option<Arc<BitVec<u64, Lsb0>>>,
    arg_cnt: usize,

    logn: u32,
    logn2: u32,
    logn4: u32,

    superblock: Vec<usize>, // position of 1st occurrence in each 4096-chunk
    blocks: Vec<Block>,     // one per superblock

    _pd: PhantomData<P>,
}

impl<P: SelectTrait> Default for SelectSupportMcl<P> {
    fn default() -> Self {
        Self {
            bv: None,
            arg_cnt: 0,
            logn: 0,
            logn2: 0,
            logn4: 0,
            superblock: Vec::new(),
            blocks: Vec::new(),
            _pd: PhantomData,
        }
    }
}

impl<P: SelectTrait> SelectSupportMcl<P> {
    pub fn new(bv: Arc<BitVec<u64, Lsb0>>) -> Self {
        let mut s = Self::default();
        s.set_vector(bv);
        s
    }

    pub fn serialize(&self, mut writer: &mut impl std::io::Write) {
        bincode::serialize_into(&mut writer, &self.arg_cnt).unwrap();
        bincode::serialize_into(&mut writer, &self.logn).unwrap();
        bincode::serialize_into(&mut writer, &self.logn2).unwrap();
        bincode::serialize_into(&mut writer, &self.logn4).unwrap();
        bincode::serialize_into(&mut writer, &(self.superblock.len() as u64)).unwrap();
        for sb in &self.superblock {
            bincode::serialize_into(&mut writer, sb).unwrap();
        }

        writer.write_all(bytemuck::cast_slice(&[self.blocks.len() as u64])).unwrap();
        self.blocks.iter().for_each(|b| b.serialize(writer));
    }

    pub fn load(mut reader: &mut impl std::io::Read, bv: Arc<BitVec<u64, Lsb0>>) -> Self {
        let arg_cnt: usize = bincode::deserialize_from(&mut reader).unwrap();
        let logn: u32 = bincode::deserialize_from(&mut reader).unwrap();
        let logn2: u32 = bincode::deserialize_from(&mut reader).unwrap();
        let logn4: u32 = bincode::deserialize_from(&mut reader).unwrap();
        let sb_len: u64 = bincode::deserialize_from(&mut reader).unwrap();

        let mut superblock = Vec::with_capacity(sb_len as usize);
        for _ in 0..sb_len {
            let sb_i: usize = bincode::deserialize_from(&mut reader).unwrap();
            superblock.push(sb_i);
        }

        let mut blocks_len_buf = [0u8; 8];
        reader.read_exact(&mut blocks_len_buf).unwrap();
        let blocks_len = u64::from_le_bytes(blocks_len_buf) as usize;

        let mut blocks = Vec::with_capacity(blocks_len);
        for _ in 0..blocks_len {
            blocks.push(Block::load(&mut reader));
        }

        Self {
            bv: Some(bv),
            arg_cnt,
            logn,
            logn2,
            logn4,
            superblock,
            blocks,
            _pd: PhantomData,
        }
    }

    pub fn set_vector(&mut self, bv: Arc<BitVec<u64, Lsb0>>) {
        self.bv = Some(bv);
        self.rebuild_slow();
    }

    pub fn occurrences(&self) -> usize {
        self.arg_cnt
    }

    /// 1-based select: select(i) = position of i-th occurrence.
    pub fn select(&self, i: usize) -> usize {
        assert!(i > 0 && i <= self.arg_cnt, "select out of range");
        let bv = self.bv.as_ref().expect("SelectSupportMcl not initialized");
        let words = bv.as_raw_slice();
        let len_bits = bv.len();

        // Convert to 0-based occurrence index
        let mut occ = i - 1;

        let sb_idx = occ >> 12; // /4096
        let offset = occ & 0xFFF;

        match &self.blocks[sb_idx] {
            Block::Long(pos) => {
                // long stores exact positions for this block (size <= 4096)
                pos[offset]
            }
            Block::Mini { mini, .. } => {
                if (offset & 0x3F) == 0 {
                    // exactly on a 64-sampled occurrence
                    self.superblock[sb_idx] + (mini[offset >> 6] as usize)
                } else {
                    // within a 64-group inside this 4096 block
                    let group = offset >> 6; // which 64-group within superblock
                    let within = offset & 0x3F; // 1..63 occurrences after the sampled one

                    // occurrences to find from the scan start:
                    // SDSL does: i = i - sb*4096 - group*64; now i in 1..64
                    let need = within as u32; // in 1..63 (since within!=0)

                    let pos = self.superblock[sb_idx] + mini[group] as usize + 1;

                    // Scan from pos for `need`-th occurrence
                    let mut word_pos = pos >> 6;
                    let word_off = (pos & 63) as u8;

                    let w0 = word_at(words, len_bits, word_pos);
                    let mut args = P::args_in_first_word(w0, word_off);
                    if args >= need {
                        return (word_pos << 6) + (P::ith_arg_pos_in_first_word(w0, need, word_off) as usize);
                    }

                    let mut sum_args = args;
                    word_pos += 1;

                    loop {
                        let w = word_at(words, len_bits, word_pos);
                        args = P::args_in_word(w);
                        if sum_args + args >= need {
                            let kth = need - sum_args;
                            return (word_pos << 6) + (P::ith_arg_pos_in_word(w, kth) as usize);
                        }
                        sum_args += args;
                        word_pos += 1;
                        // (In valid usage, we must find it before running out.)
                    }
                }
            }
        }
    }

    #[inline]
    pub fn operator(&self, i: usize) -> usize {
        self.select(i)
    }

    // ---------------- build (slow path like SDSL init_slow) ----------------

    fn rebuild_slow(&mut self) {
        let Some(bv) = &self.bv else {
            return;
        };

        self.arg_cnt = 0;
        self.superblock.clear();
        self.blocks.clear();

        let cap_bits = bv.as_raw_slice().len() * 64;
        if cap_bits == 0 {
            self.logn = 0;
            self.logn2 = 0; 
            self.logn4 = 0;
            return;
        }
        self.logn = hi(cap_bits as u64) + 1;
        self.logn2 = self.logn * self.logn;
        self.logn4 = self.logn2 * self.logn2;

        self.arg_cnt = P::arg_cnt(bv);
        if self.arg_cnt == 0 {
            return;
        }

        const SUPER: usize = 4096;
        let sb = (self.arg_cnt + SUPER - 1) / SUPER;

        self.superblock.reserve(sb);
        self.blocks.reserve(sb);

        let mut buf = [0usize; SUPER];
        let mut buf_cnt: usize = 0;
        let mut sb_cnt: usize = 0;
        let mut total_seen: usize = 0;

        for bit_i in 0..bv.len() {
            if P::found_arg(bit_i, bv) {
                buf[buf_cnt] = bit_i;
                buf_cnt += 1;
                total_seen += 1;

                if buf_cnt == SUPER || total_seen == self.arg_cnt {
                    // flush one superblock
                    let first = buf[0];
                    self.superblock.push(first);

                    let last = buf[buf_cnt - 1];
                    let diff = last - first;

                    if (diff as u32) > self.logn4 {
                        // long: store all positions
                        self.blocks.push(Block::Long(buf[..buf_cnt].to_vec()));
                    } else {
                        // mini: store every 64th relative
                        let mut mini = [0u32; 64];
                        for j in (0..buf_cnt).step_by(64) {
                            mini[j / 64] = (buf[j] - first) as u32;
                        }
                        self.blocks.push(Block::Mini { mini, cnt: buf_cnt });
                    }

                    sb_cnt += 1;
                    buf_cnt = 0;
                }
            }
        }

        debug_assert_eq!(sb_cnt, self.superblock.len());
        debug_assert_eq!(sb_cnt, self.blocks.len());
    }
}

// ============================
// helpers
// ============================

#[inline]
fn hi(x: u64) -> u32 {
    63 - x.leading_zeros()
}

#[inline]
fn low_mask(k: u32) -> u64 {
    match k {
        0 => 0,
        64 => !0u64,
        _ => (1u64 << k) - 1,
    }
}

/// Safely read a word, masking off bits beyond len_bits in the last word.
#[inline]
fn word_at(words: &[u64], len_bits: usize, widx: usize) -> u64 {
    if widx >= words.len() {
        return 0;
    }
    let mut w = words[widx];
    if widx + 1 == words.len() {
        let rem = len_bits & 63;
        if rem != 0 {
            w &= low_mask(rem as u32); // Lsb0: keep low rem bits
        }
    }
    w
}


// ============================================================
// Adapter: plug SelectSupportMcl into the BP SelectSupport trait
// ============================================================
//
// Assumes you have (from the previous port):
//   - trait SelectTrait
//   - structs SelectSupportMcl<P>, Sel1 (and optionally Sel0, Sel10, Sel01)
//
// And your BP expects this trait:
//   pub trait SelectSupport: Clone {
//       fn new(bv: Arc<BitVec<u64, Lsb0>>) -> Self;
//       fn set_vector(&mut self, bv: Arc<BitVec<u64, Lsb0>>);
//       fn select1(&self, k: usize) -> usize;
//   }

use bitvec::prelude::*;

use crate::traits::{Sel1, SelectSupport, SelectTrait};

#[derive(Clone)]
pub struct SelectSupportMclAdapter<P: SelectTrait + Default = Sel1> {
    inner: SelectSupportMcl<P>,
}

impl<P: SelectTrait + Default> SelectSupport for SelectSupportMclAdapter<P> {
    fn new(bv: Arc<BitVec<u64, Lsb0>>) -> Self {
        Self {
            inner: SelectSupportMcl::<P>::new(bv),
        }
    }

    fn set_vector(&mut self, bv: Arc<BitVec<u64, Lsb0>>) {
        self.inner.set_vector(bv);
    }

    fn select1(&self, k: usize) -> usize {
        // BP expects select1(k) = position of k-th 1-bit (k>=1)
        // MCL select is also 1-based.
        self.inner.select(k)
    }
}


// ============================
// tests (pattern 1 only, extend as needed)
// ============================

#[cfg(test)]
mod tests {
    use crate::traits::Sel1;

    use super::*;

    fn naive_select_ones(bits: &BitVec<u64, Lsb0>, k: usize) -> usize {
        let mut c = 0usize;
        for i in 0..bits.len() {
            if bits[i] {
                c += 1;
                if c == k {
                    return i;
                }
            }
        }
        panic!("k too large");
    }

    #[test]
    fn mcl_select_ones_matches_naive_small() {
        let bv: BitVec<u64, Lsb0> = bitvec![u64, Lsb0;
            1,0,1,1,0,0,1,0,  1,1,1,0,0,1,0,0
        ];
        let bv = Arc::new(bv);
        let ss = SelectSupportMcl::<Sel1>::new(bv.clone());
        let m = ss.occurrences();
        for k in 1..=m {
            assert_eq!(ss.select(k), naive_select_ones(&bv, k));
        }
    }
}

#[cfg(test)]
mod select_mcl_stress_tests {
    use crate::traits::Sel0;

    use super::*;
    use bitvec::prelude::*;
    use std::sync::Arc;

    // ---------------- deterministic PRNG (no external crates) ----------------
    #[derive(Clone)]
    struct XorShift64 {
        state: u64,
    }
    impl XorShift64 {
        fn new(seed: u64) -> Self {
            Self { state: seed.max(1) }
        }
        fn next_u64(&mut self) -> u64 {
            let mut x = self.state;
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            self.state = x;
            x
        }
        fn gen_range(&mut self, lo: usize, hi: usize) -> usize {
            debug_assert!(lo < hi);
            lo + (self.next_u64() as usize % (hi - lo))
        }
        fn gen_bool(&mut self) -> bool {
            (self.next_u64() & 1) == 1
        }
    }

    /// Build a bitvector with long alternating runs of 0s and 1s.
    /// Total length == min_len.
    fn build_runs_bv(min_len: usize, rng: &mut XorShift64) -> BitVec<u64, Lsb0> {
        let mut bv: BitVec<u64, Lsb0> = BitVec::with_capacity(min_len);
        let mut bit = rng.gen_bool();

        while bv.len() < min_len {
            // Long runs: include very long + medium + shorter (still "runny")
            let run_len = match rng.gen_range(0, 10) {
                0 => rng.gen_range(20_000, 60_000),
                1 | 2 => rng.gen_range(5_000, 20_000),
                _ => rng.gen_range(500, 5_000),
            };

            let remaining = min_len - bv.len();
            let take = run_len.min(remaining);
            bv.extend(std::iter::repeat(bit).take(take));
            bit = !bit;
        }

        bv
    }

    /// Precompute select answers for ones and zeros in a single linear pass.
    /// `ones_pos[k-1]` = position of k-th 1-bit, `zeros_pos[k-1]` = position of k-th 0-bit.
    fn precompute_select_tables(bv: &BitVec<u64, Lsb0>) -> (Vec<usize>, Vec<usize>) {
        let ones = bv.count_ones();
        let zeros = bv.len() - ones;

        let mut ones_pos = Vec::with_capacity(ones);
        let mut zeros_pos = Vec::with_capacity(zeros);

        // Linear-time pass
        for (i, b) in bv.iter().by_vals().enumerate() {
            if b {
                ones_pos.push(i);
            } else {
                zeros_pos.push(i);
            }
        }

        debug_assert_eq!(ones_pos.len(), ones);
        debug_assert_eq!(zeros_pos.len(), zeros);

        (ones_pos, zeros_pos)
    }

    #[test]
    fn select_support_mcl_stress_long_runs_100k_x10_fast_reference() {
        const CASES: usize = 10;
        const LEN: usize = 100_000;

        let mut rng = XorShift64::new(0xC0FFEE_1234_5678);

        for case in 0..CASES {
            let bv = build_runs_bv(LEN, &mut rng);
            assert_eq!(bv.len(), LEN);

            // Precompute exact answers in O(n)
            let (ones_pos, zeros_pos) = precompute_select_tables(&bv);
            let ones = ones_pos.len();
            let zeros = zeros_pos.len();

            // Ensure both symbols appear (should with alternating runs, but guard anyway)
            assert!(ones > 0, "case {case}: no ones generated");
            assert!(zeros > 0, "case {case}: no zeros generated");

            let arc = Arc::new(bv);

            // Build MCL
            let sel1 = SelectSupportMcl::<Sel1>::new(arc.clone());
            let sel0 = SelectSupportMcl::<Sel0>::new(arc.clone());

            assert_eq!(sel1.occurrences(), ones, "case {case}: ones count mismatch");
            assert_eq!(sel0.occurrences(), zeros, "case {case}: zeros count mismatch");

            // Build a good set of k's: edges + around 64/4096 boundaries + random
            let mut ks_ones: Vec<usize> = Vec::new();
            let mut ks_zeros: Vec<usize> = Vec::new();

            // Edges
            ks_ones.extend(
                [1, 2, 3, ones.saturating_sub(2), ones.saturating_sub(1), ones]
                    .into_iter()
                    .filter(|&k| (1..=ones).contains(&k)),
            );
            ks_zeros.extend(
                [1, 2, 3, zeros.saturating_sub(2), zeros.saturating_sub(1), zeros]
                    .into_iter()
                    .filter(|&k| (1..=zeros).contains(&k)),
            );

            // Superblock (4096) and miniblock (64) boundaries ±1
            for &step in &[64usize, 4096usize] {
                for mult in 1..=40 {
                    let k = mult * step;
                    for kk in [k.saturating_sub(1), k, k + 1] {
                        if (1..=ones).contains(&kk) {
                            ks_ones.push(kk);
                        }
                        if (1..=zeros).contains(&kk) {
                            ks_zeros.push(kk);
                        }
                    }
                }
            }

            // Random samples (a lot, but each check is O(1) reference now)
            for _ in 0..5000 {
                ks_ones.push(rng.gen_range(1, ones + 1));
                ks_zeros.push(rng.gen_range(1, zeros + 1));
            }

            ks_ones.sort_unstable();
            ks_ones.dedup();
            ks_zeros.sort_unstable();
            ks_zeros.dedup();

            // Validate ones
            for &k in &ks_ones {
                let got = sel1.select(k);
                let exp = ones_pos[k - 1];
                eprintln!("{} {}", exp, got);
                assert_eq!(
                    got, exp,
                    "case {case}: Sel1 select mismatch at k={k} (ones={ones})"
                );
                assert!(arc[got], "case {case}: Sel1 returned non-one at pos={got}");
            }

            // Validate zeros
            for &k in &ks_zeros {
                let got = sel0.select(k);
                let exp = zeros_pos[k - 1];
                assert_eq!(
                    got, exp,
                    "case {case}: Sel0 select mismatch at k={k} (zeros={zeros})"
                );
                assert!(!arc[got], "case {case}: Sel0 returned non-zero at pos={got}");
            }
        }
    }
}
