//! Port of sdsl::bp_support_sada (core data structure + construction) to Rust.
//!
//! Balanced parentheses bitvector convention (same as SDSL):
//!   1 = '(' (open), 0 = ')' (close)
//!
//! This implements:
//!   - building the Sadakane min/max summaries (small blocks + medium block min/max tree)
//!   - excess(), rank(), select()
//!   - fwd_excess(), bwd_excess() (used by find_close/find_open/enclose)
//!   - find_close(), find_open(), enclose()
//!
//! Notes:
//! - The original SDSL uses succinct int_vector and specialized rank/select. Here we provide
//!   simple backends (prefix sums + positions vector) so the structure is usable out-of-the-box.
//! - Indices are 0-based. select(k) expects k>=1, returning position of k-th 1-bit.
//! - bwd_excess() can return "virtual -1" in the C++ code. In Rust we expose that as `Option<isize>`
//!   where `Some(-1)` is represented as `Some(-1)` (still an isize).

//! bp_support_sada (Sadakane) port wired to:
//! - `bitvec` crate for the underlying bitvector
//! - `RankSupportV` (Vigna-style) for rank
//! - simple select backend (`SelectSupportBitvec`) for select1
//!
//! Convention: 1 = '(' (open), 0 = ')' (close)
//!
//! Public API includes: excess, rank, select, find_close, find_open, enclose
//!
//! Cargo.toml:
//!   [dependencies]
//!   bitvec = "1"

use bitvec::prelude::*;
use std::sync::Arc;

use crate::{rank_support_v::RankSupportVAdapter, select_support_mcl::SelectSupportMclAdapter, traits::{Pat1, RankSupport, RankTrait, Sel1, SelectSupport}};



// ============================================================
// bp_support_sada (Sadakane) — bitvec-based + RankSupportV plugged in
// ============================================================

#[derive(Clone, Debug)]
pub struct BpSupportSada<
    const SML: usize = 256,
    const MED_DEG: usize = 32,
    R: RankSupport = RankSupportVAdapter<Pat1>,
    S: SelectSupport = SelectSupportMclAdapter<Sel1>,
> {
    bp: Option<Arc<BitVec<u64, Lsb0>>>,
    bp_rank: Option<R>,
    bp_select: Option<S>,

    sml_block_min_max: Vec<i32>, // 2*sml_blocks
    med_block_min_max: Vec<i64>, // 2*(med_blocks + med_inner_blocks)

    size: usize,
    sml_blocks: usize,
    med_blocks: usize,
    med_inner_blocks: usize,
}

impl<const SML: usize, const MED_DEG: usize, R: RankSupport, S: SelectSupport>
    BpSupportSada<SML, MED_DEG, R, S>
{
    pub fn new() -> Self {
        Self {
            bp: None,
            bp_rank: None,
            bp_select: None,
            sml_block_min_max: Vec::new(),
            med_block_min_max: Vec::new(),
            size: 0,
            sml_blocks: 0,
            med_blocks: 0,
            med_inner_blocks: 0,
        }
    }

    pub fn build(bp: Arc<BitVec<u64, Lsb0>>) -> Self {
        assert!(SML > 0);

        let size = bp.len();
        if size == 0 {
            return Self::new();
        }

        let sml_blocks = (size + SML - 1) / SML;
        let med_block_len = SML * MED_DEG;
        let med_blocks = (size + med_block_len - 1) / med_block_len;

        let mut med_inner_blocks = 1usize;
        while med_inner_blocks < med_blocks {
            med_inner_blocks <<= 1;
            assert!(med_inner_blocks != 0);
        }
        med_inner_blocks -= 1;

        let mut out = Self {
            bp: Some(bp.clone()),
            bp_rank: Some(R::new(bp.clone())),
            bp_select: Some(S::new(bp.clone())),
            sml_block_min_max: vec![0; 2 * sml_blocks],
            med_block_min_max: vec![0; 2 * (med_blocks + med_inner_blocks)],
            size,
            sml_blocks,
            med_blocks,
            med_inner_blocks,
        };

        // Build small block summaries + medium leaf summaries
        let mut min_ex: i64 = 1;
        let mut max_ex: i64 = -1;
        let mut curr_rel_ex: i64 = 0;
        let mut curr_abs_ex: i64 = 0;

        for i in 0..size {
            if out.bp()[i] {
                curr_rel_ex += 1;
            } else {
                curr_rel_ex -= 1;
            }
            if curr_rel_ex > max_ex {
                max_ex = curr_rel_ex;
            }
            if curr_rel_ex < min_ex {
                min_ex = curr_rel_ex;
            }

            let end_of_sml = (i + 1) % SML == 0 || (i + 1) == size;
            if end_of_sml {
                let sidx = i / SML;

                // Store transformed min/max like SDSL
                out.sml_block_min_max[2 * sidx] = -((min_ex - 1) as i32);
                out.sml_block_min_max[2 * sidx + 1] = (max_ex + 1) as i32;

                // Medium leaf index
                let v = out.med_inner_blocks + (sidx / MED_DEG);

                let cand_min_store = -(curr_abs_ex + min_ex) + (out.size as i64);
                let cand_max_store = (curr_abs_ex + max_ex) + (out.size as i64);

                if cand_min_store > out.med_block_min_max[2 * v] {
                    out.med_block_min_max[2 * v] = cand_min_store;
                }
                if cand_max_store > out.med_block_min_max[2 * v + 1] {
                    out.med_block_min_max[2 * v + 1] = cand_max_store;
                }

                curr_abs_ex += curr_rel_ex;

                min_ex = 1;
                max_ex = -1;
                curr_rel_ex = 0;
            }
        }

        // Build medium internal nodes bottom-up
        let nodes = out.med_blocks + out.med_inner_blocks;
        if nodes > 0 {
            for v in (1..nodes).rev() {
                let p = parent(v);
                if out.min_value(v) < out.min_value(p) {
                    out.med_block_min_max[2 * p] = out.med_block_min_max[2 * v];
                }
                if out.max_value(v) > out.max_value(p) {
                    out.med_block_min_max[2 * p + 1] = out.med_block_min_max[2 * v + 1];
                }
            }
        }

        out
    }

    pub fn set_vector(&mut self, bp: Arc<BitVec<u64, Lsb0>>) {
        self.size = bp.len();
        self.bp = Some(bp.clone());

        if let Some(r) = &mut self.bp_rank {
            r.set_vector(bp.clone());
        } else {
            self.bp_rank = Some(R::new(bp.clone()));
        }

        if let Some(s) = &mut self.bp_select {
            s.set_vector(bp);
        } else {
            self.bp_select = Some(S::new(bp));
        }
    }

    pub fn len(&self) -> usize {
        self.size
    }

    #[inline]
    fn bp(&self) -> &BitVec<u64, Lsb0> {
        self.bp.as_ref().expect("BpSupportSada not initialized")
    }
    #[inline]
    fn rank_backend(&self) -> &R {
        self.bp_rank.as_ref().expect("BpSupportSada not initialized")
    }
    #[inline]
    fn select_backend(&self) -> &S {
        self.bp_select.as_ref().expect("BpSupportSada not initialized")
    }

    /// excess(i) = 2*rank1(i+1) - (i+1)
    pub fn excess(&self, i: usize) -> i64 {
        let r = self.rank_backend().rank1(i + 1) as i64;
        (r << 1) - (i as i64) - 1
    }

    /// excess before position `pos` (prefix excluding pos). Safe for pos=0 -> 0.
    #[inline]
    fn excess_before(&self, pos: usize) -> i64 {
        if pos == 0 {
            0
        } else {
            self.excess(pos - 1)
        }
    }

    /// rank(i) = #opens in [0..=i]
    pub fn rank(&self, i: usize) -> usize {
        self.rank_backend().rank1(i + 1)
    }

    /// select(k) = position of k-th open, k>=1
    pub fn select(&self, k: usize) -> usize {
        self.select_backend().select1(k)
    }

    /// find_close(i): if i is '(', return matching ')', else i
    pub fn find_close(&self, i: usize) -> usize {
        assert!(i < self.size);
        if !self.bp()[i] {
            return i;
        }
        self.fwd_excess(i, -1).unwrap_or(self.size)
    }

    /// find_open(i): if i is ')', return matching '(', else i
    pub fn find_open(&self, i: usize) -> usize {
        assert!(i < self.size);
        if self.bp()[i] {
            return i;
        }
        // if excess(j)==excess(i) for maximal j<i, then open is j+1
        match self.bwd_excess(i, 0) {
            Some(j) if j >= 0 => (j as usize) + 1,
            Some(-1) => 0,
            _ => self.size,
        }
    }

    /// enclose(i): opening paren of smallest pair that strictly encloses i, else n
    pub fn enclose(&self, i: usize) -> usize {
        assert!(i < self.size);

        if !self.bp()[i] {
            let o = self.find_open(i);
            return if o < self.size { self.enclose(o) } else { self.size };
        }

        // For an open at i (depth d = excess(i)), parent open is at (j+1) where
        // j is the last position < i with excess(j) == d-2. j can be -1 => parent is 0.
        match self.bwd_excess(i, -2) {
            Some(j) if j >= -1 => (j + 1) as usize,
            _ => self.size,
        }
    }

    // ---------------- Sadakane min/max helpers ----------------

    #[inline]
    fn sml_block_idx(i: usize) -> usize {
        i / SML
    }
    #[inline]
    fn med_block_idx(i: usize) -> usize {
        i / (SML * MED_DEG)
    }

    #[inline]
    fn is_root(v: usize) -> bool {
        v == 0
    }
    #[inline]
    fn is_left_child(v: usize) -> bool {
        v % 2 == 1
    }
    #[inline]
    fn is_right_child(v: usize) -> bool {
        v % 2 == 0
    }
    #[inline]
    fn is_leaf(&self, v: usize) -> bool {
        v >= self.med_inner_blocks
    }

    #[inline]
    fn min_value(&self, v: usize) -> i64 {
        (self.size as i64) - self.med_block_min_max[2 * v]
    }
    #[inline]
    fn max_value(&self, v: usize) -> i64 {
        self.med_block_min_max[2 * v + 1] - (self.size as i64)
    }
    #[inline]
    fn sml_min_value(&self, b: usize) -> i64 {
        1 - (self.sml_block_min_max[2 * b] as i64)
    }
    #[inline]
    fn sml_max_value(&self, b: usize) -> i64 {
        (self.sml_block_min_max[2 * b + 1] as i64) - 1
    }

    // ---------------- near scans (<=SML) ----------------

    fn near_fwd_excess(&self, start: usize, desired_excess: i64) -> Option<usize> {
        let end = usize::min(self.size, start + SML);
        for j in start..end {
            if self.excess(j) == desired_excess {
                return Some(j);
            }
        }
        None
    }

    fn near_bwd_excess(&self, start: isize, desired_excess: i64) -> Option<isize> {
        let mut steps = 0usize;
        let mut j = start;
        while j >= 0 && steps < SML {
            if self.excess(j as usize) == desired_excess {
                return Some(j);
            }
            j -= 1;
            steps += 1;
        }
        if j < 0 && desired_excess == 0 {
            return Some(-1);
        }
        None
    }

    // ---------------- in-medium-block scans (fixed) ----------------

    fn fwd_excess_in_med_block(&self, mut b: usize, desired_excess: i64) -> Option<usize> {
        let med = b / MED_DEG;
        let mut end_b = (med + 1) * MED_DEG;
        if end_b > self.sml_blocks {
            end_b = self.sml_blocks;
        }

        while b < end_b {
            let block_start = b * SML;
            if block_start >= self.size {
                break;
            }

            let base = self.excess_before(block_start);
            let min_ex = base + self.sml_min_value(b);
            let max_ex = base + self.sml_max_value(b);

            if min_ex <= desired_excess && desired_excess <= max_ex {
                let end = usize::min(self.size, block_start + SML);
                for j in block_start..end {
                    if self.excess(j) == desired_excess {
                        return Some(j);
                    }
                }
            }
            b += 1;
        }
        None
    }

    fn bwd_excess_in_med_block(&self, mut b: usize, desired_excess: i64) -> Option<usize> {
        let first_b = (b / MED_DEG) * MED_DEG;

        loop {
            let block_start = b * SML;
            if block_start < self.size {
                let base = self.excess_before(block_start);
                let min_ex = base + self.sml_min_value(b);
                let max_ex = base + self.sml_max_value(b);

                if min_ex <= desired_excess && desired_excess <= max_ex {
                    let end = usize::min(self.size, block_start + SML);
                    for j in (block_start..end).rev() {
                        if self.excess(j) == desired_excess {
                            return Some(j);
                        }
                    }
                }
            }

            if b == first_b {
                break;
            }
            b -= 1;
        }

        None
    }

    // ---------------- tree guided search ----------------

    /// minimal j>i with excess(j)=excess(i)+rel
    fn fwd_excess(&self, i: usize, rel: i64) -> Option<usize> {
        let desired = self.excess(i) + rel;

        // (1) near scan within small block
        if let Some(j) = self.near_fwd_excess(i + 1, desired) {
            return Some(j);
        }

        // (2) scan remaining small blocks in current medium block
        if let Some(j) = self.fwd_excess_in_med_block(Self::sml_block_idx(i) + 1, desired) {
            return Some(j);
        }

        // (3) tree climb/down across medium blocks
        if Self::med_block_idx(i) >= self.med_blocks {
            return None;
        }
        let mut v = self.med_inner_blocks + Self::med_block_idx(i);

        // go up
        while !Self::is_root(v) {
            if Self::is_left_child(v) {
                v += 1; // right sibling
                if self.min_value(v) <= desired && desired <= self.max_value(v) {
                    break;
                }
            }
            v = parent(v);
        }

        if !Self::is_root(v) {
            // go down
            while !self.is_leaf(v) {
                v = left_child(v);
                if !(self.min_value(v) <= desired && desired <= self.max_value(v)) {
                    v += 1;
                }
            }
            let first_sml = (v - self.med_inner_blocks) * MED_DEG;
            return self.fwd_excess_in_med_block(first_sml, desired);
        }

        None
    }

    /// maximal j<i with excess(j)=excess(i)+rel, can return -1 sentinel (virtual)
    fn bwd_excess(&self, i: usize, rel: i64) -> Option<isize> {
        let desired = self.excess(i) + rel;

        if i == 0 {
            return if desired == 0 { Some(-1) } else { None };
        }

        // (1) near scan
        if let Some(j) = self.near_bwd_excess((i as isize) - 1, desired) {
            return Some(j);
        }

        // (2) scan previous small blocks within current medium block (fixed underflow + sentinel)
        let sml_i = Self::sml_block_idx(i);
        if let Some(prev) = sml_i.checked_sub(1) {
            if let Some(j) = self.bwd_excess_in_med_block(prev, desired) {
                return Some(j as isize);
            }
        } else if desired == 0 {
            return Some(-1);
        }

        // (3) tree climb/down across medium blocks
        if Self::med_block_idx(i) == 0 {
            return if desired == 0 { Some(-1) } else { None };
        }

        let mut v = self.med_inner_blocks + Self::med_block_idx(i);

        // go up
        while !Self::is_root(v) {
            if Self::is_right_child(v) {
                v -= 1; // left sibling
                if self.min_value(v) <= desired && desired <= self.max_value(v) {
                    break;
                }
            }
            v = parent(v);
        }

        if !Self::is_root(v) {
            // go down
            while !self.is_leaf(v) {
                v = right_child(v);
                if !(self.min_value(v) <= desired && desired <= self.max_value(v)) {
                    v -= 1;
                }
            }
            let last_sml = (v - self.med_inner_blocks) * MED_DEG + (MED_DEG - 1);
            return self
                .bwd_excess_in_med_block(last_sml, desired)
                .map(|x| x as isize)
                .or_else(|| if desired == 0 { Some(-1) } else { None });
        }

        if desired == 0 {
            Some(-1)
        } else {
            None
        }
    }
}

// Heap navigation (SDSL layout)
#[inline]
fn parent(v: usize) -> usize {
    (v - 1) / 2
}
#[inline]
fn left_child(v: usize) -> usize {
    2 * v + 1
}
#[inline]
fn right_child(v: usize) -> usize {
    2 * v + 2
}

// ============================================================
// Suggested default alias for “BP + RankSupportV + bitvec”
// ============================================================

pub type BpSupportSadaBitvec<const SML: usize = 256, const MED_DEG: usize = 32> =
    BpSupportSada<SML, MED_DEG, RankSupportVAdapter<Pat1>, SelectSupportMclAdapter<Sel1>>;


#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn bitvec_from_bools(bits: &[bool]) -> BitVec<u64, Lsb0> {
        let mut bv: BitVec<u64, Lsb0> = BitVec::with_capacity(bits.len());
        bv.extend(bits.iter().copied());
        bv
    }

    /// Generate all balanced parenthesis sequences (Dyck words) of length 2*n.
    /// `true` = '(' , `false` = ')'
    fn gen_balanced(n: usize, mut f: impl FnMut(&[bool])) {
        fn rec(
            n: usize,
            open: usize,
            close: usize,
            buf: &mut Vec<bool>,
            f: &mut dyn FnMut(&[bool]),
        ) {
            if buf.len() == 2 * n {
                f(buf);
                return;
            }
            if open < n {
                buf.push(true);
                rec(n, open + 1, close, buf, f);
                buf.pop();
            }
            if close < open {
                buf.push(false);
                rec(n, open, close + 1, buf, f);
                buf.pop();
            }
        }

        let mut buf = Vec::with_capacity(2 * n);
        rec(n, 0, 0, &mut buf, &mut f);
    }

    #[derive(Debug)]
    struct NaiveRef {
        // match arrays
        open_to_close: Vec<usize>, // for open positions, where it closes; otherwise usize::MAX
        close_to_open: Vec<usize>, // for close positions, where it opens; otherwise usize::MAX
        // enclose(open) => parent open, or n if none (only meaningful for opens, but we’ll define for all)
        enclose_open: Vec<usize>,
        // excess(i)
        excess: Vec<i64>,
        // rank(i) = #opens in [0..=i]
        rank: Vec<usize>,
        // select(k) = pos of k-th open, k>=1
        select: Vec<usize>,
    }

    fn build_naive(bits: &[bool]) -> NaiveRef {
        let n = bits.len();
        let mut open_to_close = vec![usize::MAX; n];
        let mut close_to_open = vec![usize::MAX; n];
        let mut enclose_open = vec![n; n];

        // For enclose: parent of an open is the stack top *before* pushing that open.
        let mut stack: Vec<usize> = Vec::new();

        // For matches: on close, pop.
        for i in 0..n {
            if bits[i] {
                // '('
                let parent = stack.last().copied().unwrap_or(n);
                enclose_open[i] = parent;
                stack.push(i);
            } else {
                // ')'
                let open = stack.pop().expect("sequence must be balanced");
                open_to_close[open] = i;
                close_to_open[i] = open;
            }
        }
        assert!(stack.is_empty(), "sequence must be balanced");

        // excess + rank + select
        let mut excess = vec![0i64; n];
        let mut rank = vec![0usize; n];
        let mut select = Vec::new();
        let mut e: i64 = 0;
        let mut r: usize = 0;
        for i in 0..n {
            if bits[i] {
                e += 1;
                r += 1;
                select.push(i);
            } else {
                e -= 1;
            }
            excess[i] = e;
            rank[i] = r;
        }

        NaiveRef {
            open_to_close,
            close_to_open,
            enclose_open,
            excess,
            rank,
            select,
        }
    }

    fn enclose_naive(refs: &NaiveRef, bits: &[bool], i: usize) -> usize {
        let n = bits.len();
        if bits[i] {
            // open
            refs.enclose_open[i]
        } else {
            // close: same as its matching open
            let o = refs.close_to_open[i];
            if o == usize::MAX {
                n
            } else {
                refs.enclose_open[o]
            }
        }
    }

    #[test]
    fn exhaustive_bp_support_sada_up_to_20() {
        // We deliberately choose small parameters to stress block boundaries and tree logic.
        // (The defaults SML=256, MED_DEG=32 are fine, but less likely to hit edge cases for n<=20.)
        type DS = BpSupportSadaBitvec<4, 2>;

        for pairs in 0..=10 {
            let len = 2 * pairs;

            gen_balanced(pairs, |bits| {
                assert_eq!(bits.len(), len);

                let naive = build_naive(bits);

                let bv = Arc::new(bitvec_from_bools(bits));
                let ds = DS::build(bv);

                // Basic length
                assert_eq!(ds.len(), len);

                if len == 0 {
                    return;
                }

                // excess, rank, find_close/open, enclose for all positions
                for i in 0..len {
                    // excess(i)
                    assert_eq!(
                        ds.excess(i),
                        naive.excess[i],
                        "excess mismatch at i={i}, bits={:?}",
                        bits
                    );

                    // rank(i)
                    assert_eq!(
                        ds.rank(i),
                        naive.rank[i],
                        "rank mismatch at i={i}, bits={:?}",
                        bits
                    );

                    // find_close / find_open
                    if bits[i] {
                        // open
                        let exp_close = naive.open_to_close[i];
                        assert_ne!(exp_close, usize::MAX);
                        assert_eq!(
                            ds.find_close(i),
                            exp_close,
                            "find_close mismatch at i={i}, bits={:?}",
                            bits
                        );
                        // find_open(open) must be itself
                        assert_eq!(
                            ds.find_open(i),
                            i,
                            "find_open(open) mismatch at i={i}, bits={:?}",
                            bits
                        );
                    } else {
                        // close
                        let exp_open = naive.close_to_open[i];
                        assert_ne!(exp_open, usize::MAX);
                        assert_eq!(
                            ds.find_open(i),
                            exp_open,
                            "find_open mismatch at i={i}, bits={:?}",
                            bits
                        );
                        // find_close(close) must be itself
                        assert_eq!(
                            ds.find_close(i),
                            i,
                            "find_close(close) mismatch at i={i}, bits={:?}",
                            bits
                        );
                    }

                    // enclose
                    let exp_enclose = enclose_naive(&naive, bits, i);
                    assert_eq!(
                        ds.enclose(i),
                        exp_enclose,
                        "enclose mismatch at i={i}, bits={:?}",
                        bits
                    );
                }

                // select(k) for all opens (k>=1)
                for (k0, &pos) in naive.select.iter().enumerate() {
                    let k = k0 + 1;
                    assert_eq!(
                        ds.select(k),
                        pos,
                        "select mismatch at k={k}, bits={:?}",
                        bits
                    );
                }
            });
        }
    }

    // ---- deterministic PRNG (no external crates) ----
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
        fn next_usize(&mut self) -> usize {
            self.next_u64() as usize
        }
    }

    /// Generate a random Dyck word of `pairs` pairs by uniformly shuffling
    /// among all valid next moves (not uniform over Catalan objects, but fine for testing).
    fn random_balanced(pairs: usize, rng: &mut XorShift64) -> Vec<bool> {
        let n = 2 * pairs;
        let mut out = Vec::with_capacity(n);
        let mut open = 0usize;
        let mut close = 0usize;

        while out.len() < n {
            let can_open = open < pairs;
            let can_close = close < open;

            match (can_open, can_close) {
                (true, true) => {
                    // choose randomly
                    if (rng.next_u64() & 1) == 0 {
                        out.push(true);
                        open += 1;
                    } else {
                        out.push(false);
                        close += 1;
                    }
                }
                (true, false) => {
                    out.push(true);
                    open += 1;
                }
                (false, true) => {
                    out.push(false);
                    close += 1;
                }
                (false, false) => break,
            }
        }
        out
    }

    #[test]
    fn random_bp_support_sada_len_1000_100_cases() {
        // Stress small blocks / medium blocks even for long sequences
        type DS = BpSupportSadaBitvec<64, 8>;

        let pairs = 500;
        let len = 2 * pairs;

        let mut rng = XorShift64::new(0xD1CE_BAAD_F00D_F00D);

        for case in 0..100 {
            let bits = random_balanced(pairs, &mut rng);
            assert_eq!(bits.len(), len);

            let naive = build_naive(&bits);
            let bv = Arc::new(bitvec_from_bools(&bits));
            let ds = DS::build(bv);

            // Check all per-position queries
            for i in 0..len {
                assert_eq!(
                    ds.excess(i),
                    naive.excess[i],
                    "case {case}: excess mismatch at i={i}"
                );

                assert_eq!(
                    ds.rank(i),
                    naive.rank[i],
                    "case {case}: rank mismatch at i={i}"
                );

                if bits[i] {
                    // open
                    let exp_close = naive.open_to_close[i];
                    assert_ne!(exp_close, usize::MAX);
                    assert_eq!(
                        ds.find_close(i),
                        exp_close,
                        "case {case}: find_close mismatch at i={i}"
                    );
                    assert_eq!(
                        ds.find_open(i),
                        i,
                        "case {case}: find_open(open) mismatch at i={i}"
                    );
                } else {
                    // close
                    let exp_open = naive.close_to_open[i];
                    assert_ne!(exp_open, usize::MAX);
                    assert_eq!(
                        ds.find_open(i),
                        exp_open,
                        "case {case}: find_open mismatch at i={i}"
                    );
                    assert_eq!(
                        ds.find_close(i),
                        i,
                        "case {case}: find_close(close) mismatch at i={i}"
                    );
                }

                let exp_enclose = enclose_naive(&naive, &bits, i);
                assert_eq!(
                    ds.enclose(i),
                    exp_enclose,
                    "case {case}: enclose mismatch at i={i}"
                );
            }

            // Check select for all opens
            assert_eq!(naive.select.len(), pairs);
            for (k0, &pos) in naive.select.iter().enumerate() {
                let k = k0 + 1;
                assert_eq!(
                    ds.select(k),
                    pos,
                    "case {case}: select mismatch at k={k}"
                );
            }

            // Optional: spot-check a few random positions again (redundant but nice)
            for _ in 0..25 {
                let i = rng.next_usize() % len;
                if bits[i] {
                    assert_eq!(ds.find_close(i), naive.open_to_close[i], "case {case}: spot find_close");
                } else {
                    assert_eq!(ds.find_open(i), naive.close_to_open[i], "case {case}: spot find_open");
                }
                assert_eq!(ds.enclose(i), enclose_naive(&naive, &bits, i), "case {case}: spot enclose");
            }
        }
    }

}

