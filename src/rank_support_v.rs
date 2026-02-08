use bitvec::prelude::*;
use std::{marker::PhantomData, sync::Arc};
use serde::{Serialize, Deserialize};

use crate::{traits::{Pat1, RankSupport, RankTrait}, util::word_at};

/// Port of `sdsl::rank_support_v`:
/// - superblock 512 bits (8 words)
/// - stores per-superblock absolute + packed relative counts (7×9-bit, in one u64)
#[derive(Debug, Clone)]
pub struct RankSupportV<T: RankTrait = Pat1> {
    bv: Option<Arc<BitVec<u64, Lsb0>>>,
    len_bits: usize,
    basic_block: Vec<u64>,
    _pd: PhantomData<T>,
}

impl<T: RankTrait> Default for RankSupportV<T> {
    fn default() -> Self {
        Self {
            bv: None,
            len_bits: 0,
            basic_block: Vec::new(),
            _pd: PhantomData,
        }
    }
}

/// Adapter so RankSupportV can be used as BP's RankSupport.
#[derive(Clone)]
pub struct RankSupportVAdapter<T: RankTrait + Default = Pat1> {
    inner: crate::rank_support_v::RankSupportV<T>,
}

impl<T: RankTrait + Default> RankSupport for RankSupportVAdapter<T> {
    fn new(bv: Arc<BitVec<u64, Lsb0>>) -> Self {
        Self {
            inner: crate::rank_support_v::RankSupportV::<T>::new(bv),
        }
    }
    fn set_vector(&mut self, bv: Arc<BitVec<u64, Lsb0>>) {
        self.inner.set_vector(bv);
    }
    fn rank1(&self, i: usize) -> usize {
        self.inner.rank(i)
    }
}

impl<T: RankTrait> RankSupportV<T> {
    pub fn new(bv: Arc<BitVec<u64, Lsb0>>) -> Self {
        let mut rs = Self::default();
        rs.set_vector(bv);
        rs
    }

    pub fn set_vector(&mut self, bv: Arc<BitVec<u64, Lsb0>>) {
        self.len_bits = bv.len();
        self.bv = Some(bv);
        self.rebuild();
    }

    pub fn get_vector(&self) -> &Option<Arc<BitVec<u64, Lsb0>>> {
        &self.bv
    }

    pub fn serialize(&self, mut writer: &mut impl std::io::Write) {
        bincode::serialize_into(&mut writer, &self.len_bits).unwrap();
        bincode::serialize_into(writer, &self.basic_block).unwrap();
    }

    pub fn load(mut reader: &mut impl std::io::Read, bv: Arc<BitVec<u64, Lsb0>>) -> Self {
        let len_bits: usize = bincode::deserialize_from(&mut reader).unwrap();
        let basic_block: Vec<u64> = bincode::deserialize_from(&mut reader).unwrap();
        Self {
            bv: Some(bv),
            len_bits,
            basic_block,
            _pd: PhantomData,
        }
    }

    /// occurrences in prefix [0..idx)
    pub fn rank(&self, idx: usize) -> usize {
        assert!(idx <= self.len_bits);
        if self.len_bits == 0 {
            return 0;
        }
        let bv = self.bv.as_ref().expect("RankSupportV not initialized");
        let words = bv.as_raw_slice();

        let p = (idx >> 8) & !1; // (idx/512)*2
        let super_abs = self.basic_block[p];
        let rel_pack = self.basic_block[p + 1];

        let b = ((idx & 0x1FF) >> 6) as u64; // 0..7
        let rel = if b == 0 {
            0
        } else {
            (rel_pack >> (63 - 9 * b)) & 0x1FF
        };

        let mut ans = super_abs + rel;
        if (idx & 63) != 0 {
            ans += T::word_rank(words, self.len_bits, idx) as u64;
        }
        ans as usize
    }

    /// occurrences in prefix [0..idx)
    pub unsafe fn rank_unchecked(&self, idx: usize) -> usize {
        unsafe {
            let bv = self.bv.as_ref().unwrap();
            let words = bv.as_raw_slice();

            let p = (idx >> 8) & !1; // (idx/512)*2
            let super_abs = self.basic_block.get_unchecked(p);
            let rel_pack = self.basic_block.get_unchecked(p + 1);

            let b = ((idx & 0x1FF) >> 6) as u64; // 0..7
            let rel = if b == 0 {
                0
            } else {
                (rel_pack >> (63 - 9 * b)) & 0x1FF
            };

            let mut ans = super_abs + rel;
            if (idx & 63) != 0 {
                ans += T::word_rank(words, self.len_bits, idx) as u64;
            }
            ans as usize
        }
    }

    fn rebuild(&mut self) {
        let Some(bv) = &self.bv else {
            return;
        };
        if self.len_bits == 0 {
            self.basic_block = vec![0, 0];
            return;
        }

        let words = bv.as_raw_slice();
        let cap_bits = words.len() * 64;
        let basic_block_size = (((cap_bits >> 9) + 1) << 1) as usize; // ((cap/512)+1)*2
        self.basic_block.clear();
        self.basic_block.resize(basic_block_size, 0);

        let mut j: usize = 0;
        self.basic_block[0] = 0;
        self.basic_block[1] = 0;

        let mut carry = T::init_carry();
        let mut sum: u64 = if !words.is_empty() {
            T::args_in_word(word_at(words, self.len_bits, 0), &mut carry) as u64
        } else {
            0
        };

        let mut second_level_cnt: u64 = 0;

        for i in 1..words.len() {
            if (i & 0x7) == 0 {
                j += 2;
                self.basic_block[j - 1] = second_level_cnt;
                self.basic_block[j] = self.basic_block[j - 2] + sum;
                second_level_cnt = 0;
                sum = 0;
            } else {
                second_level_cnt |= sum << (63 - 9 * (i as u64 & 0x7));
            }

            let w = word_at(words, self.len_bits, i);
            sum += T::args_in_word(w, &mut carry) as u64;
        }

        let i = words.len();
        if (i & 0x7) != 0 {
            second_level_cnt |= sum << (63 - 9 * ((i as u64) & 0x7));
            if j + 1 < self.basic_block.len() {
                self.basic_block[j + 1] = second_level_cnt;
            }
        } else {
            j += 2;
            if j >= 2 && j < self.basic_block.len() {
                self.basic_block[j - 1] = second_level_cnt;
                self.basic_block[j] = self.basic_block[j - 2] + sum;
            }
            if j + 1 < self.basic_block.len() {
                self.basic_block[j + 1] = 0;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::traits::{Pat0, Pat1};

    use super::*;

    fn naive_rank1(bits: &BitVec<u64, Lsb0>, idx: usize) -> usize {
        bits[..idx].count_ones()
    }

    #[test]
    fn rank_support_v_matches_naive_pat1_small() {
        let mut bv: BitVec<u64, Lsb0> = bitvec![u64, Lsb0;];
        bv.extend([true, false, true, true, false, false, true, false].iter().copied());
        let bv = Arc::new(bv);

        let rs = RankSupportV::<Pat1>::new(bv.clone());
        for i in 0..=bv.len() {
            assert_eq!(rs.rank(i), naive_rank1(&bv, i), "idx={i}");
        }
    }

    #[test]
    fn rank_support_v_matches_naive_pat0_small() {
        let mut bv: BitVec<u64, Lsb0> = bitvec![u64, Lsb0;];
        bv.extend([true, false, true, true, false, false, true, false].iter().copied());
        let bv = Arc::new(bv);

        let rs = RankSupportV::<Pat0>::new(bv.clone());
        for i in 0..=bv.len() {
            let naive = i - bv[..i].count_ones();
            assert_eq!(rs.rank(i), naive, "idx={i}");
        }
    }
}
