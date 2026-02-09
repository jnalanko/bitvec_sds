
// ============================================================
// rank_support_v (ported, bitvec-based)
// ============================================================

use std::sync::Arc;
use bitvec::prelude::*;

use crate::util::{low_mask, select_in_word, word_at};

pub trait RankTrait: Clone + Default {
    fn init_carry() -> u64;
    fn args_in_word(w: u64, carry: &mut u64) -> u32;
    fn word_rank(words: &[u64], len_bits: usize, idx: usize) -> u32;
}

// ============================
// Pattern trait (like SDSL select_support_trait)
// ============================

pub trait SelectTrait: Clone + Default {
    /// Total occurrences in the whole bitvector.
    fn arg_cnt(bv: &BitVec<u64, Lsb0>) -> usize;

    /// Whether an occurrence is "found" at absolute bit index `i`.
    fn found_arg(i: usize, bv: &BitVec<u64, Lsb0>) -> bool;

    /// #occurrences in first word starting from bit offset `word_off` (0..63), given carry-in.
    fn args_in_first_word(word: u64, word_off: u8) -> u32;

    /// position (0..63) of i-th occurrence in first word starting at offset, given carry-in.
    /// `i` is 1-based within this word-range.
    fn ith_arg_pos_in_first_word(word: u64, i: u32, word_off: u8) -> u32;

    /// #occurrences in a full word, updating carry to carry-out.
    fn args_in_word(word: u64) -> u32;

    /// position (0..63) of i-th occurrence in the word (1-based), given carry-in.
    fn ith_arg_pos_in_word(word: u64, i: u32) -> u32;
}

// ============================================================
// BP rank/select traits (generic over backends)
// ============================================================

pub trait RankSupport: Clone {
    fn new(bv: Arc<BitVec<u64, Lsb0>>) -> Self;
    fn set_vector(&mut self, bv: Arc<BitVec<u64, Lsb0>>);
    /// #ones in [0, i)
    fn rank1(&self, i: usize) -> usize;
}

pub trait SelectSupport: Clone {
    fn new(bv: Arc<BitVec<u64, Lsb0>>) -> Self;
    fn set_vector(&mut self, bv: Arc<BitVec<u64, Lsb0>>);
    /// position of k-th 1 (k>=1)
    fn select1(&self, k: usize) -> usize;
}


/// Pattern: `1` (rank1)
#[derive(Clone, Default, Debug)]
pub struct Pat1;
impl RankTrait for Pat1 {
    #[inline]
    fn init_carry() -> u64 {
        0
    }
    #[inline]
    fn args_in_word(w: u64, _carry: &mut u64) -> u32 {
        w.count_ones()
    }
    #[inline]
    fn word_rank(words: &[u64], len_bits: usize, idx: usize) -> u32 {
        let widx = idx >> 6;
        let k = (idx & 63) as u32;
        if k == 0 {
            return 0;
        }
        let w = word_at(words, len_bits, widx);
        (w & low_mask(k)).count_ones()
    }
}

/// Pattern: `0` (rank0) — use RankSupportV<Pat0>
#[derive(Clone, Default, Debug)]
pub struct Pat0;
impl RankTrait for Pat0 {
    #[inline]
    fn init_carry() -> u64 {
        0
    }
    #[inline]
    fn args_in_word(w: u64, _carry: &mut u64) -> u32 {
        (!w).count_ones()
    }
    #[inline]
    fn word_rank(words: &[u64], len_bits: usize, idx: usize) -> u32 {
        let widx = idx >> 6;
        let k = (idx & 63) as u32;
        if k == 0 {
            return 0;
        }
        let w = word_at(words, len_bits, widx);
        ((!w) & low_mask(k)).count_ones()
    }
}

// -------- Pattern: 1 --------

#[derive(Clone, Default, Debug)]
pub struct Sel1;

impl SelectTrait for Sel1 {
    fn arg_cnt(bv: &BitVec<u64, Lsb0>) -> usize {
        bv.count_ones()
    }

    fn found_arg(i: usize, bv: &BitVec<u64, Lsb0>) -> bool {
        bv[i]
    }

    fn args_in_first_word(word: u64, word_off: u8) -> u32 {
        let mask = !low_mask(word_off as u32);
        (word & mask).count_ones()
    }

    fn ith_arg_pos_in_first_word(word: u64, i: u32, word_off: u8) -> u32 {
        let mask = !low_mask(word_off as u32);
        select_in_word(word & mask, i)
    }

    fn args_in_word(word: u64) -> u32 {
        word.count_ones()
    }

    fn ith_arg_pos_in_word(word: u64, i: u32) -> u32 {
        select_in_word(word, i)
    }
}

// -------- Pattern: 0 --------

#[derive(Clone, Default, Debug)]
pub struct Sel0;

impl SelectTrait for Sel0 {
    fn arg_cnt(bv: &BitVec<u64, Lsb0>) -> usize {
        bv.len() - bv.count_ones()
    }

    fn found_arg(i: usize, bv: &BitVec<u64, Lsb0>) -> bool {
        !bv[i]
    }

    fn args_in_first_word(word: u64, word_off: u8) -> u32 {
        let mask = !low_mask(word_off as u32);
        ((!word) & mask).count_ones()
    }

    fn ith_arg_pos_in_first_word(word: u64, i: u32, word_off: u8) -> u32 {
        let mask = !low_mask(word_off as u32);
        select_in_word((!word) & mask, i)
    }

    fn args_in_word(word: u64) -> u32 {
        (!word).count_ones()
    }

    fn ith_arg_pos_in_word(word: u64, i: u32) -> u32 {
        select_in_word(!word, i)
    }
}

pub fn count_args_by_words<P: SelectTrait>(bv: &BitVec<u64, Lsb0>) -> usize {
    let words = bv.as_raw_slice();
    let len_bits = bv.len();

    if len_bits == 0 {
        return 0;
    }

    let mut sum: u64 = 0;
    for wi in 0..words.len() {
        let w = word_at(words, len_bits, wi);
        sum += P::args_in_word(w) as u64;
    }
    sum as usize
}

/// Random-access u32 sequence used for construction.
/// Must be stable across calls (same len + same values for same indices).
pub trait RandomAccessU32 {
    fn len(&self) -> usize;
    fn get(&self, idx: usize) -> u32;
}

// Allow passing references easily (&T where T: RandomAccessU32).
impl<T: RandomAccessU32 + ?Sized> RandomAccessU32 for &T {
    #[inline]
    fn len(&self) -> usize {
        (*self).len()
    }
    #[inline]
    fn get(&self, idx: usize) -> u32 {
        (*self).get(idx)
    }
}

// Common impls
impl RandomAccessU32 for [u32] {
    #[inline]
    fn len(&self) -> usize {
        <[u32]>::len(self)
    }
    #[inline]
    fn get(&self, idx: usize) -> u32 {
        self[idx]
    }
}

impl RandomAccessU32 for Vec<u32> {
    #[inline]
    fn len(&self) -> usize {
        self.len()
    }
    #[inline]
    fn get(&self, idx: usize) -> u32 {
        self[idx]
    }
}

impl RandomAccessU32 for Box<[u32]> {
    #[inline]
    fn len(&self) -> usize {
        self.as_ref().len()
    }
    #[inline]
    fn get(&self, idx: usize) -> u32 {
        self.as_ref()[idx]
    }
}

impl RandomAccessU32 for Arc<[u32]> {
    #[inline]
    fn len(&self) -> usize {
        self.as_ref().len()
    }
    #[inline]
    fn get(&self, idx: usize) -> u32 {
        self.as_ref()[idx]
    }
}