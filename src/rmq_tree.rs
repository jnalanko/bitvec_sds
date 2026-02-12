use bitvec::prelude::*;

use crate::traits::{IntArray, RandomAccessU64};

/// A compact array storing fixed-width integers in a bitvector.
/// Each element uses `bits_per_element` bits.
pub struct PackedArray {
    data: BitVec<usize, Lsb0>,
    len: usize,
    bits_per_element: usize,
}

impl PackedArray {
    fn new(bits_per_element: usize) -> Self {
        assert!(bits_per_element > 0 && bits_per_element <= 64);
        Self {
            data: BitVec::new(),
            len: 0,
            bits_per_element,
        }
    }

    fn with_values(values: &impl RandomAccessU64, bits_per_element: usize) -> Self {
        assert!(bits_per_element > 0 && bits_per_element <= 64);
        let len = values.len();
        let mut data = BitVec::with_capacity(len * bits_per_element);
        for i in 0..len {
            let v = values.get(i);
            debug_assert!(
                bits_per_element == 64 || v < (1u64 << bits_per_element),
                "value {v} does not fit in {bits_per_element} bits"
            );
            for b in 0..bits_per_element {
                data.push((v >> b) & 1 != 0);
            }
        }
        Self {
            data,
            len,
            bits_per_element,
        }
    }

    fn len(&self) -> usize {
        self.len
    }

    fn get(&self, index: usize) -> u64 {
        debug_assert!(index < self.len);
        let start = index * self.bits_per_element;
        let mut val: u64 = 0;
        for b in 0..self.bits_per_element {
            if self.data[start + b] {
                val |= 1u64 << b;
            }
        }
        val
    }

    fn push(&mut self, value: u64) {
        debug_assert!(
            self.bits_per_element == 64 || value < (1u64 << self.bits_per_element),
            "value {value} does not fit in {} bits",
            self.bits_per_element
        );
        for b in 0..self.bits_per_element {
            self.data.push((value >> b) & 1 != 0);
        }
        self.len += 1;
    }
}

/// Minimum number of bits needed to represent `max_val` (unsigned).
/// Returns 1 for max_val == 0 or 1.
fn bits_for_max(max_val: u64) -> usize {
    if max_val <= 1 {
        return 1;
    }
    64 - max_val.leading_zeros() as usize
}

impl IntArray for PackedArray {
    fn with_values(values: &impl RandomAccessU64, bits_per_element: usize) -> Self {
        PackedArray::with_values(values, bits_per_element)
    }

    fn get(&self, index: usize) -> u64 {
        PackedArray::get(self, index)
    }
}

/// A heap-type segment tree holding block minima, supporting RMQ, PSV, and NSV
/// queries. The underlying array `x` and the tree nodes are stored in
/// bit-packed form using `ceil(log2(max_value))` bits per element.
///
/// Ported from rmq_tree.h (Kempa & Puglisi).
pub struct RmqTree<A: IntArray = PackedArray> {
    /// The original sequence, bit-packed.
    x: A,
    /// Segment tree over block minima, bit-packed (1-indexed, size = 2*tree_size).
    tree: A,
    /// Number of elements.
    n: usize,
    /// Half the tree array length (number of leaves in the complete binary tree).
    tree_size: usize,
    /// Block size exponent: each block has `1 << block_bits` elements.
    block_bits: usize,
    /// Block size = `1 << block_bits`.
    block_len: usize,
    /// Block mask = `block_len - 1`.
    block_mask: usize,
}

impl<A: IntArray> RmqTree<A> {
    /// Build an `RmqTree` over `values` with the given block-size exponent.
    ///
    /// * `block_bits` – each block has `2^block_bits` elements.
    /// * All values must be non-negative.
    pub fn new(values: &impl RandomAccessU64, block_bits: usize) -> Self {
        let n = values.len();
        assert!(n > 0, "values must be non-empty");

        let block_len: usize = 1 << block_bits;
        let block_mask: usize = block_len - 1;

        // Determine bits per element from the maximum value (and the sentinel `n`).
        let mut max_val = n as u64;
        for i in 0..n {
            max_val = max_val.max(values.get(i));
        }
        let bpe = bits_for_max(max_val);

        let x = A::with_values(values, bpe);

        // Compute tree_size: smallest power of two such that tree_size * block_len >= n.
        let mut tree_size: usize = 1;
        while (tree_size << block_bits) < n {
            tree_size <<= 1;
        }

        // Build tree: indices [tree_size .. 2*tree_size) are leaves.
        // Sentinel value for empty leaves is `n` (larger than any real value).
        let sentinel = n as u64;
        let total = tree_size << 1;
        let mut tree_vals: Vec<u64> = vec![sentinel; total];

        // Fill leaves with block minima.
        let mut block_idx = 0usize;
        let mut i = 0usize;
        while i < n {
            let end = (i + block_len).min(n);
            let mut bmin = values.get(i);
            for k in (i + 1)..end {
                bmin = bmin.min(values.get(k));
            }
            tree_vals[tree_size + block_idx] = bmin;
            block_idx += 1;
            i += block_len;
        }

        // Build inner nodes bottom-up.
        for i in (1..tree_size).rev() {
            tree_vals[i] = tree_vals[2 * i].min(tree_vals[2 * i + 1]);
        }

        let tree = A::with_values(&tree_vals.as_slice(), bpe);

        Self {
            x,
            tree,
            n,
            tree_size,
            block_bits,
            block_len,
            block_mask,
        }
    }

    /// Return the minimum value in `x[i..=j]`.
    pub fn rmq(&self, mut i: usize, mut j: usize) -> u64 {
        debug_assert!(i <= j && j < self.n);

        let mut m = self.x.get(i);

        // Small range: linear scan.
        if j - i <= 512 {
            for k in (i + 1)..=j {
                m = m.min(self.x.get(k));
            }
            return m;
        }

        // Scan to a block boundary from the left.
        while (i & self.block_mask) != 0 && i <= j {
            m = m.min(self.x.get(i));
            i += 1;
        }
        // Scan to a block boundary from the right.
        while ((j + 1) & self.block_mask) != 0 && i <= j {
            m = m.min(self.x.get(j));
            j -= 1;
        }

        if i > j {
            return m;
        }

        // Map to tree leaves.
        i = self.tree_size + (i >> self.block_bits);
        j = self.tree_size + (j >> self.block_bits);

        let mut klo = i;
        let mut klen: usize = 1;
        let mut k = i;
        m = m.min(self.tree.get(i));

        // Walk up the tree.
        while klo + klen <= j {
            if k % 2 == 1 {
                klo -= klen;
            } else {
                if klo + 2 * klen - 1 <= j {
                    m = m.min(self.tree.get(k + 1));
                }
            }
            klen *= 2;
            k >>= 1;
        }

        // Walk down the tree towards j.
        while j != klo + klen - 1 {
            let left = 2 * k;
            let right = 2 * k + 1;
            klen /= 2;

            if j >= klo + klen {
                if klo >= i {
                    m = m.min(self.tree.get(left));
                }
                klo += klen;
                k = right;
            } else {
                k = left;
            }
        }
        if klo >= i {
            m = m.min(self.tree.get(k));
        }

        m
    }

    /// Return the largest `j < i` such that `x[j] < ub`, or `None`.
    pub fn psv(&self, mut i: usize, ub: u64) -> Option<usize> {
        // Linear scan nearby.
        let start = i;
        while self.x.get(i) >= ub && start - i < 512 {
            if i == 0 {
                return None;
            }
            i -= 1;
        }
        if self.x.get(i) < ub {
            return Some(i);
        }
        if i == 0 {
            return None;
        }

        // Scan to a block boundary.
        let mut j = i - 1;
        while (j + 1) & self.block_mask != 0 {
            if self.x.get(j) < ub {
                return Some(j);
            }
            if j == 0 {
                return None;
            }
            j -= 1;
        }

        // Walk up the tree looking for a left neighbor block with min < ub.
        let mut ti = self.tree_size + (i >> self.block_bits);
        while ti != 1 {
            if (ti & 1) != 0 && self.tree.get(ti - 1) < ub {
                ti -= 1;
                break;
            }
            ti >>= 1;
        }
        if ti == 1 {
            return None;
        }

        // Walk down to find the rightmost qualifying block.
        while ti < self.tree_size {
            ti = (ti << 1) + if self.tree.get(2 * ti + 1) < ub { 1 } else { 0 };
        }

        // Scan that block right-to-left.
        let block_start = (ti - self.tree_size) << self.block_bits;
        let block_end = (block_start + self.block_len).min(self.n) - 1;
        let mut k = block_end;
        loop {
            if self.x.get(k) < ub {
                return Some(k);
            }
            if k == block_start {
                break;
            }
            k -= 1;
        }
        None
    }

    /// Return the smallest `j > i` such that `x[j] < ub`, or `None`.
    pub fn nsv(&self, mut i: usize, ub: u64) -> Option<usize> {
        // Linear scan nearby.
        let start = i;
        while i < self.n && self.x.get(i) >= ub && i - start < 512 {
            i += 1;
        }
        if i < self.n && self.x.get(i) < ub {
            return Some(i);
        }
        if i >= self.n {
            return None;
        }

        // Scan to a block boundary.
        let mut j = i + 1;
        while j < self.n && (j & self.block_mask) != 0 {
            if self.x.get(j) < ub {
                return Some(j);
            }
            j += 1;
        }

        // Walk up the tree looking for a right neighbor block with min < ub.
        let mut ti = self.tree_size + (i >> self.block_bits);
        while ti != 1 {
            if (ti & 1) == 0 && self.tree.get(ti + 1) < ub {
                ti += 1;
                break;
            }
            ti >>= 1;
        }
        if ti == 1 {
            return None;
        }

        // Walk down to find the leftmost qualifying block.
        while ti < self.tree_size {
            ti = (ti << 1) + if self.tree.get(2 * ti) >= ub { 1 } else { 0 };
        }

        // Scan that block left-to-right.
        let block_start = (ti - self.tree_size) << self.block_bits;
        let block_end = (block_start + self.block_len).min(self.n);
        for k in block_start..block_end {
            if self.x.get(k) < ub {
                return Some(k);
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── PackedArray tests ──────────────────────────────────────────────

    #[test]
    fn packed_array_roundtrip() {
        let vals: Vec<u64> = vec![0, 1, 2, 3, 15, 7, 8, 14];
        let pa = PackedArray::with_values(&vals.as_slice(), 4);
        assert_eq!(pa.len(), vals.len());
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(pa.get(i), v);
        }
    }

    #[test]
    fn packed_array_push() {
        let mut pa = PackedArray::new(8);
        for v in 0..=255u64 {
            pa.push(v);
        }
        assert_eq!(pa.len(), 256);
        for v in 0..=255u64 {
            assert_eq!(pa.get(v as usize), v);
        }
    }

    #[test]
    fn bits_for_max_cases() {
        assert_eq!(bits_for_max(0), 1);
        assert_eq!(bits_for_max(1), 1);
        assert_eq!(bits_for_max(2), 2);
        assert_eq!(bits_for_max(3), 2);
        assert_eq!(bits_for_max(4), 3);
        assert_eq!(bits_for_max(255), 8);
        assert_eq!(bits_for_max(256), 9);
    }

    // ── RMQ tests ──────────────────────────────────────────────────────

    fn brute_rmq(vals: &[u64], i: usize, j: usize) -> u64 {
        vals[i..=j].iter().copied().min().unwrap()
    }

    #[test]
    fn rmq_small() {
        let vals: Vec<u64> = vec![5, 3, 7, 1, 4, 6, 2, 8];
        let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), 1); // block size = 2
        for i in 0..vals.len() {
            for j in i..vals.len() {
                assert_eq!(
                    tree.rmq(i, j),
                    brute_rmq(&vals, i, j),
                    "rmq({i},{j})"
                );
            }
        }
    }

    #[test]
    fn rmq_single_element() {
        let vals: Vec<u64> = vec![42];
        let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), 2);
        assert_eq!(tree.rmq(0, 0), 42);
    }

    #[test]
    fn rmq_all_same() {
        let vals: Vec<u64> = vec![5; 20];
        let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), 2);
        assert_eq!(tree.rmq(0, 19), 5);
        assert_eq!(tree.rmq(3, 15), 5);
    }

    #[test]
    fn rmq_large_exhaustive() {
        // Use a non-trivial sequence and block size 3 (block_len = 8).
        let vals: Vec<u64> = (0..200)
            .map(|i| ((i * 37 + 13) % 100) as u64)
            .collect();
        let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), 3);
        // Test a selection of ranges.
        for i in (0..vals.len()).step_by(7) {
            for j in (i..vals.len()).step_by(11) {
                assert_eq!(
                    tree.rmq(i, j),
                    brute_rmq(&vals, i, j),
                    "rmq({i},{j})"
                );
            }
        }
    }

    #[test]
    fn rmq_spanning_many_blocks() {
        // 1024 elements, block size 4 (block_len=16), query the full range.
        let vals: Vec<u64> = (0..1024).rev().collect();
        let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), 4);
        assert_eq!(tree.rmq(0, 1023), 0);
        assert_eq!(tree.rmq(0, 500), vals[500]);
        assert_eq!(tree.rmq(500, 1023), 0);
    }

    // ── PSV tests ──────────────────────────────────────────────────────

    fn brute_psv(vals: &[u64], i: usize, ub: u64) -> Option<usize> {
        (0..i).rev().find(|&j| vals[j] < ub)
    }

    #[test]
    fn psv_basic() {
        let vals: Vec<u64> = vec![5, 3, 7, 1, 4, 6, 2, 8];
        let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), 1);
        for i in 0..vals.len() {
            let ub = vals[i];
            assert_eq!(
                tree.psv(i, ub),
                brute_psv(&vals, i, ub),
                "psv({i}, {ub})"
            );
        }
    }

    #[test]
    fn psv_no_match() {
        let vals: Vec<u64> = vec![10, 20, 30, 40];
        let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), 1);
        // No element < 10 exists before index 3.
        assert_eq!(tree.psv(3, 10), None);
        // No element < 1 anywhere.
        assert_eq!(tree.psv(3, 1), None);
    }

    #[test]
    fn psv_exhaustive() {
        let vals: Vec<u64> = (0..150)
            .map(|i| ((i * 37 + 13) % 100) as u64)
            .collect();
        let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), 2);
        for i in 0..vals.len() {
            let ub = vals[i];
            assert_eq!(
                tree.psv(i, ub),
                brute_psv(&vals, i, ub),
                "psv({i}, {ub})"
            );
        }
    }

    // ── NSV tests ──────────────────────────────────────────────────────

    fn brute_nsv(vals: &[u64], i: usize, ub: u64) -> Option<usize> {
        ((i + 1)..vals.len()).find(|&j| vals[j] < ub)
    }

    #[test]
    fn nsv_basic() {
        let vals: Vec<u64> = vec![5, 3, 7, 1, 4, 6, 2, 8];
        let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), 1);
        for i in 0..vals.len() {
            let ub = vals[i];
            assert_eq!(
                tree.nsv(i, ub),
                brute_nsv(&vals, i, ub),
                "nsv({i}, {ub})"
            );
        }
    }

    #[test]
    fn nsv_no_match() {
        let vals: Vec<u64> = vec![10, 20, 30, 40];
        let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), 1);
        assert_eq!(tree.nsv(0, 10), None);
        assert_eq!(tree.nsv(3, 1), None);
    }

    #[test]
    fn nsv_exhaustive() {
        let vals: Vec<u64> = (0..150)
            .map(|i| ((i * 37 + 13) % 100) as u64)
            .collect();
        let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), 2);
        for i in 0..vals.len() {
            let ub = vals[i];
            assert_eq!(
                tree.nsv(i, ub),
                brute_nsv(&vals, i, ub),
                "nsv({i}, {ub})"
            );
        }
    }

    // ── Mixed / edge-case tests ────────────────────────────────────────

    #[test]
    fn descending_sequence() {
        // [63, 62, 61, ..., 0]
        let vals: Vec<u64> = (0..64).rev().collect();
        let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), 2);
        // All previous elements are larger, so PSV should be None.
        for i in 0..64 {
            assert_eq!(tree.psv(i, vals[i]), None, "psv({i})");
        }
        // Each next element is smaller, so NSV should find it.
        for i in 0..63 {
            assert_eq!(tree.nsv(i, vals[i]), Some(i + 1), "nsv({i})");
        }
        assert_eq!(tree.nsv(63, vals[63]), None);
    }

    #[test]
    fn ascending_sequence() {
        // [0, 1, 2, ..., 63]
        let vals: Vec<u64> = (0..64).collect();
        let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), 2);
        // Each previous element is smaller, so PSV finds it.
        assert_eq!(tree.psv(0, vals[0]), None);
        for i in 1..64 {
            assert_eq!(tree.psv(i, vals[i]), Some(i - 1), "psv({i})");
        }
        // All later elements are larger, so NSV should be None.
        for i in 0..64 {
            assert_eq!(tree.nsv(i, vals[i]), None, "nsv({i})");
        }
    }

    #[test]
    fn various_block_sizes() {
        let vals: Vec<u64> = (0..300)
            .map(|i| ((i * 53 + 7) % 200) as u64)
            .collect();
        for block_bits in 1..=5 {
            let tree = RmqTree::<PackedArray>::new(&vals.as_slice(), block_bits);
            // Spot-check a few RMQs.
            assert_eq!(tree.rmq(0, vals.len() - 1), *vals.iter().min().unwrap());
            assert_eq!(tree.rmq(10, 50), brute_rmq(&vals, 10, 50));
            assert_eq!(tree.rmq(100, 299), brute_rmq(&vals, 100, 299));
        }
    }
}
