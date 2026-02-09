use std::ops::Range;
use std::sync::Arc;

use bitvec::prelude::*;
use crate::traits::*;
use crate::rank_support_v::RankSupportV;
use crate::select_support_mcl::SelectSupportMcl;

/// Rank support: # of 1-bits in B[0..idx)
pub trait RankSupport {
    fn rank1(&self, idx: usize) -> usize;

    #[inline]
    fn rank0(&self, idx: usize) -> usize {
        idx - self.rank1(idx)
    }

    fn serialize(&self, writer: &mut impl std::io::Write);
    fn load(reader: &mut impl std::io::Read, bv: Arc<BitVec<u64, Lsb0>>) -> Self;
}

impl RankSupport for RankSupportV<Pat1> {
    fn rank1(&self, idx: usize) -> usize {
        self.rank(idx)
    }
    fn serialize(&self, writer: &mut impl std::io::Write) {
        RankSupportV::<Pat1>::serialize(self, writer);
    }
    fn load(reader: &mut impl std::io::Read, bv: Arc<BitVec<u64, Lsb0>>) -> Self {
        RankSupportV::<Pat1>::load(reader, bv)
    }
}

/// Select support: position of k-th bit (1-based k), returning 0-based position.
pub trait SelectSupport {
    fn select1(&self, k: usize) -> Option<usize>;
    fn select0(&self, k: usize) -> Option<usize>;

    fn serialize(&self, writer: &mut impl std::io::Write);
    fn load(reader: &mut impl std::io::Read, bv: Arc<BitVec<u64, Lsb0>>) -> Self;
}

#[derive(Debug, Clone)]
pub struct SelectBinarySearchOverRank {
    pub rs: RankSupportV<Pat1>,
}

impl SelectSupport for SelectBinarySearchOverRank {
    fn select1(&self, k: usize) -> Option<usize> {
        let mut pos = 0;
        let n = self.rs.get_vector().as_ref().unwrap().len();

        if self.rs.rank1(n) < k { return None } // k is larger than the number of ones
        
        let mut step = n.next_power_of_two();
        // We are looking for the largest position such that rank is smaller than k
        while step > 0 {
            if pos + step < n && self.rs.rank1(pos + step) < k {
                pos += step;
            }
            step /= 2;
        }
        Some(pos)
    }

    fn select0(&self, k: usize) -> Option<usize> {
        let mut pos = 0;
        let n = self.rs.get_vector().as_ref().unwrap().len();

        if self.rs.rank0(n) < k { return None } // k is larger than the number of zeros
        
        let mut step = n.next_power_of_two();
        // We are looking for the largest position such that rank is smaller than k
        while step > 0 {
            if pos + step < n && self.rs.rank0(pos + step) < k {
                pos += step;
            }
            step /= 2;
        }
        Some(pos)
    }

    fn serialize(&self, mut writer: &mut impl std::io::Write) {
        self.rs.serialize(&mut writer);
    }

    fn load (reader: &mut impl std::io::Read, bv: Arc<BitVec<u64, Lsb0>>) -> Self {
        let rs = RankSupportV::load(reader, bv);
        Self { rs }
    }
}


// Both select 0 and select 1
#[derive(Debug, Clone)]
pub struct SelectSupportBoth {
    pub ss0: SelectSupportMcl::<Sel0>,
    pub ss1: SelectSupportMcl::<Sel1>,
}

impl SelectSupportBoth {
    pub fn new(bv: Arc<BitVec<u64, Lsb0>>) -> Self {
        Self {
            ss0: SelectSupportMcl::<Sel0>::new(bv.clone()),
            ss1: SelectSupportMcl::<Sel1>::new(bv.clone()),
        }
    }
}

impl SelectSupport for SelectSupportBoth {

    fn select0(&self, k: usize) -> Option<usize> {
        Some(self.ss0.select(k)) // Panics if k is larger than the number of 0-bits
    }

    fn select1(&self, k: usize) -> Option<usize> {
        Some(self.ss1.select(k)) // Panics if k is larger than the number of 1-bits
    }

    fn serialize(&self, writer: &mut impl std::io::Write) {
        self.ss0.serialize(writer);
        self.ss1.serialize(writer);
    }

    fn load(reader: &mut impl std::io::Read, bv: Arc<BitVec<u64, Lsb0>>) -> Self {
        let ss0 = SelectSupportMcl::<Sel0>::load(reader, bv.clone());
        let ss1 = SelectSupportMcl::<Sel1>::load(reader, bv.clone());
        Self { ss0, ss1 }
    }
}




#[derive(Debug, Clone)]
struct Node<R, S> {
    lo: u32,
    hi: u32, // exclusive
    mid: u32,
    bits: Arc<BitVec<u64, Lsb0>>,
    rank: R,
    sel: S,
    left: u32,
    right: u32,
}

#[derive(Debug, Clone)]
struct NodeBuilder {
    lo: u32,
    hi: u32, // exclusive
    mid: u32,
    bits: BitVec<u64, Lsb0>,
    left: u32,
    right: u32,
}

fn build_template(nodes: &mut Vec<NodeBuilder>, lo: u32, hi: u32) -> usize {
    let mid = lo + (hi - lo) / 2;

    let idx = nodes.len();
    nodes.push(NodeBuilder {
        lo,
        hi,
        mid,
        bits: BitVec::new(),
        left: u32::MAX,
        right: u32::MAX,
    });

    if hi - lo > 1 {
        let left = build_template(nodes, lo, mid);
        let right = build_template(nodes, mid, hi);
        nodes[idx].left = left as u32;
        nodes[idx].right = right as u32;
    }
    idx
}


#[derive(Debug, Clone)]
pub struct WaveletTree<R, S> {
    nodes: Vec<Node<R, S>>,
    n: usize,
    lo: u32,
    hi: u32, // exclusive
}

impl<R, S> WaveletTree<R, S>
where
    R: RankSupport,
    S: SelectSupport,
{

    /// Build a wavelet tree for values in [lo, hi).
    ///
    /// `data` is any random-access structure providing len() and get(i)->u32.
    pub fn new<D, FR, FS>(
        data: D,
        lo: u32,
        hi: u32,
        mut build_rank: FR,
        mut build_sel: FS,
    ) -> Self
    where
        D: RandomAccessU32,
        FR: FnMut(Arc<BitVec<u64, Lsb0>>) -> R,
        FS: FnMut(Arc<BitVec<u64, Lsb0>>) -> S,
    {
        assert!(lo < hi, "alphabet range must be non-empty");

        let n = data.len();

        // Build topology (independent of data order/length).
        let k_nodes_guess: usize = ((hi - lo) as usize).saturating_mul(2).saturating_sub(1);
        let mut builders = Vec::with_capacity(k_nodes_guess);
        let root = build_template(&mut builders, lo, hi);
        debug_assert_eq!(root, 0);

        // Pass 1: validate + count how many bits each internal node will store.
        let mut counts = vec![0usize; builders.len()];
        for i in 0..n {
            let v = data.get(i);
            assert!(v >= lo && v < hi, "value {v} out of range [{lo},{hi})");

            let mut node_idx = 0usize;
            loop {
                let node = &builders[node_idx];
                if node.hi - node.lo == 1 {
                    break; // leaf
                }
                counts[node_idx] += 1;
                node_idx = if v >= node.mid {
                    node.right as usize
                } else {
                    node.left as usize
                };
            }
        }

        // Allocate exact bitvector capacities.
        for (i, b) in builders.iter_mut().enumerate() {
            if b.hi - b.lo > 1 {
                b.bits = BitVec::with_capacity(counts[i]);
            }
        }

        // Pass 2: fill bitvectors by pushing bits along the root-to-leaf path.
        for i in 0..n {
            let v = data.get(i);

            let mut node_idx = 0usize;
            loop {
                // Copy out the routing fields to avoid borrow issues.
                let (lo_i, hi_i, mid_i, left_i, right_i) = {
                    let node = &builders[node_idx];
                    (node.lo, node.hi, node.mid, node.left, node.right)
                };

                if hi_i - lo_i == 1 {
                    break; // leaf
                }

                let go_right = v >= mid_i;
                builders[node_idx].bits.push(go_right);

                node_idx = if go_right {
                    right_i as usize
                } else {
                    left_i as usize
                };
            }
        }

        // Finalize into Node<R,S> with rank/select supports.
        let mut nodes = Vec::with_capacity(builders.len());
        for b in builders {
            let bits = Arc::new(b.bits);
            let rank = build_rank(bits.clone());
            let sel = build_sel(bits.clone());
            nodes.push(Node {
                lo: b.lo,
                hi: b.hi,
                mid: b.mid,
                bits,
                rank,
                sel,
                left: b.left,
                right: b.right,

            });
        }

        Self { nodes, n, lo, hi }
    }

    pub fn serialize(&self, mut writer: &mut impl std::io::Write) {
        bincode::serialize_into(&mut writer, &self.n).unwrap();
        bincode::serialize_into(&mut writer, &self.lo).unwrap();
        bincode::serialize_into(&mut writer, &self.hi).unwrap();
        bincode::serialize_into(&mut writer, &(self.nodes.len() as u64)).unwrap();
        for node in &self.nodes {
            bincode::serialize_into(&mut writer, &node.lo).unwrap();
            bincode::serialize_into(&mut writer, &node.hi).unwrap();
            bincode::serialize_into(&mut writer, &node.mid).unwrap();
            bincode::serialize_into(&mut writer, &(*node.bits)).unwrap();
            node.rank.serialize(&mut writer);
            node.sel.serialize(&mut writer);
            bincode::serialize_into(&mut writer, &(node.left as u64)).unwrap();
            bincode::serialize_into(&mut writer, &(node.right as u64)).unwrap();

        }
    }

    pub fn load(mut reader: &mut impl std::io::Read) -> Self {
        let n: usize = bincode::deserialize_from(&mut reader).unwrap();
        let lo: u32 = bincode::deserialize_from(&mut reader).unwrap();
        let hi: u32 = bincode::deserialize_from(&mut reader).unwrap();
        let nodes_len: u64 = bincode::deserialize_from(&mut reader).unwrap();

        let mut nodes = Vec::with_capacity(nodes_len as usize);
        for _ in 0..nodes_len {
            let lo: u32 = bincode::deserialize_from(&mut reader).unwrap();
            let hi: u32 = bincode::deserialize_from(&mut reader).unwrap();
            let mid: u32 = bincode::deserialize_from(&mut reader).unwrap();
            let bits: BitVec<u64, Lsb0> = bincode::deserialize_from(&mut reader).unwrap();
            let bits = Arc::new(bits);
            let rank = R::load(&mut reader, bits.clone());
            let sel = S::load(&mut reader, bits.clone());

            let left_raw: u64 = bincode::deserialize_from(&mut reader).unwrap();
            let left = if left_raw == u64::MAX { u32::MAX } else { left_raw as u32 };

            let right_raw: u64 = bincode::deserialize_from(&mut reader).unwrap();
            let right = if right_raw == u64::MAX { u32::MAX } else { right_raw as u32 };

            nodes.push(Node {
                lo,
                hi,
                mid,
                bits,
                rank,
                sel,
                left,
                right,
            });
        }

        Self { nodes, n, lo, hi }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.n
    }

    /// Next position > i with value < x in O(log k).
    pub fn next_smaller(&self, i: usize, x: u32) -> Option<usize> {
        if self.n == 0 || i + 1 >= self.n {
            return None;
        }
        if x <= self.lo {
            return None;
        }
        // Search in suffix [i+1, n)
        self.next_in_value_prefix_range(0, i + 1, self.n, x)
    }

    /// Previous position < i with value < x in O(log k).
    pub fn prev_smaller(&self, i: usize, x: u32) -> Option<usize> {
        if self.n == 0 || i == 0 {
            return None;
        }
        if x <= self.lo {
            return None;
        }
        // Search in prefix [0, i)
        self.prev_in_value_prefix_range(0, 0, i, x)
    }

    fn next_in_value_prefix_range(
        &self,
        node_idx: usize,
        l: usize,
        r: usize,
        x: u32,
    ) -> Option<usize> {
        if l >= r {
            return None;
        }

        let node = &self.nodes[node_idx];

        if x <= node.lo {
            return None;
        }
        if x >= node.hi {
            return Some(l);
        }

        if node.hi - node.lo == 1 {
            return Some(l);
        }

        let l1 = node.rank.rank1(l);
        let r1 = node.rank.rank1(r);
        let l0 = l - l1;
        let r0 = r - r1;

        if x < node.mid {
            // Right-child values are all >= mid > x, so only recurse left.
            self.next_in_value_prefix_range(node.left as usize, l0, r0, x)
                .and_then(|p| node.sel.select0(p + 1))
        } else if l0 < r0 {
            // x >= mid: every left-child value is < mid <= x.
            // The first 0-bit in [l,r) is a candidate with no recursion.
            let cl = node.sel.select0(l0 + 1).unwrap();
            if cl == l {
                return Some(cl); // earliest possible position, can't be beaten
            }
            // Positions [l, cl) are all 1-bits → right child [l1, l1+cl-l).
            // Any right-child hit maps to a position < cl, so it always wins.
            let narrow_r1 = l1 + cl - l;
            match self
                .next_in_value_prefix_range(node.right as usize, l1, narrow_r1, x)
                .and_then(|p| node.sel.select1(p + 1))
            {
                Some(cr) => Some(cr),
                None => Some(cl),
            }
        } else {
            // No zeros in [l, r), all elements go right.
            self.next_in_value_prefix_range(node.right as usize, l1, r1, x)
                .and_then(|p| node.sel.select1(p + 1))
        }
    }

    fn prev_in_value_prefix_range(
        &self,
        node_idx: usize,
        l: usize,
        r: usize,
        x: u32,
    ) -> Option<usize> {
        if l >= r {
            return None;
        }

        let node = &self.nodes[node_idx];

        if x <= node.lo {
            return None;
        }
        if x >= node.hi {
            return Some(r - 1);
        }

        if node.hi - node.lo == 1 {
            return Some(r - 1);
        }

        let l1 = node.rank.rank1(l);
        let r1 = node.rank.rank1(r);
        let l0 = l - l1;
        let r0 = r - r1;

        if x < node.mid {
            // Right-child values are all >= mid > x, so only recurse left.
            self.prev_in_value_prefix_range(node.left as usize, l0, r0, x)
                .and_then(|p| node.sel.select0(p + 1))
        } else if l0 < r0 {
            // x >= mid: every left-child value is < mid <= x.
            // The last 0-bit in [l,r) is a candidate with no recursion.
            let cl = node.sel.select0(r0).unwrap();
            if cl == r - 1 {
                return Some(cl); // latest possible position, can't be beaten
            }
            // Positions (cl, r) are all 1-bits → right child [cl+1-r0, r1).
            // Any right-child hit maps to a position > cl, so it always wins.
            let narrow_l1 = cl + 1 - r0;
            match self
                .prev_in_value_prefix_range(node.right as usize, narrow_l1, r1, x)
                .and_then(|p| node.sel.select1(p + 1))
            {
                Some(cr) => Some(cr),
                None => Some(cl),
            }
        } else {
            // No zeros in [l, r), all elements go right.
            self.prev_in_value_prefix_range(node.right as usize, l1, r1, x)
                .and_then(|p| node.sel.select1(p + 1))
        }
    }

    pub fn access(&self, i: usize) -> u32 {
        assert!(i < self.n, "index out of bounds: {i} >= {}", self.n);

        let mut node_idx = 0usize; // root is always 0 in this construction
        let mut pos = i;

        loop {
            let node = &self.nodes[node_idx];

            // Leaf: interval size 1 => the symbol is node.lo.
            if node.hi - node.lo == 1 {
                return node.lo;
            }

            // Compute rank1 once; derive rank0 = pos - r1
            let r1 = node.rank.rank1(pos);
            if *node.bits.get(pos).unwrap() {
                pos = r1;
                node_idx = node.right as usize;
            } else {
                pos = pos - r1;
                node_idx = node.left as usize;
            }
        }
    }

    pub fn range_min(&self, l: usize, r: usize) -> Option<u32> {
        if l >= r || r > self.n {
            return None;
        }

        let mut node_idx = 0usize; // root
        let mut lcur = l;
        let mut rcur = r;

        loop {
            let node = &self.nodes[node_idx];

            // Leaf: interval size 1 => all values here are node.lo
            if node.hi - node.lo == 1 {
                return Some(node.lo);
            }

            // Compute rank1 once per endpoint; derive rank0
            let l1 = node.rank.rank1(lcur);
            let r1 = node.rank.rank1(rcur);
            let l0 = lcur - l1;
            let r0 = rcur - r1;

            if l0 < r0 {
                // There is at least one element on the left => min must be in left
                node_idx = node.left as usize;
                lcur = l0;
                rcur = r0;
            } else {
                // Left is empty in this range => go right
                node_idx = node.right as usize;
                lcur = l1;
                rcur = r1;
            }
        }
    }

    /// Number of occurrences of value `x` in data[0..i) in O(log k).
    pub fn rank(&self, x: u32, i: usize) -> usize {
        if i == 0 || i > self.n || x < self.lo || x >= self.hi {
            return 0;
        }

        let mut node_idx = 0usize; // root
        let mut pref = i;

        loop {
            let node = &self.nodes[node_idx];

            // If x not in this node's value interval, nothing matches.
            if x < node.lo || x >= node.hi {
                return 0;
            }

            // Leaf: interval size 1 => all remaining are equal to x.
            if node.hi - node.lo == 1 {
                return pref;
            }

            if x < node.mid {
                // Go left, count zeros
                pref = node.rank.rank0(pref);
                node_idx = node.left as usize;
            } else {
                // Go right, count ones
                pref = node.rank.rank1(pref);
                node_idx = node.right as usize;
            }

            if pref == 0 {
                return 0;
            }
        }
    }

    pub fn range_rank(&self, x: u32, l: usize, r: usize) -> usize {
        if l >= r {
            return 0;
        }
        self.rank(x, r).saturating_sub(self.rank(x, l))
    }

    pub fn value_range(&self) -> Range<usize> {
        self.lo as usize .. self.hi as usize
    }

    /// Return the two smallest distinct values in `data[l..r)` in O(log k).
    pub fn range_bottom2(&self, l: usize, r: usize) -> (Option<u32>, Option<u32>) {
        if l >= r || r > self.n {
            return (None, None);
        }
        self.range_bottom2_rec(0, l, r)
    }

    fn range_bottom2_rec(
        &self,
        node_idx: usize,
        l: usize,
        r: usize,
    ) -> (Option<u32>, Option<u32>) {
        if l >= r {
            return (None, None);
        }

        let node = &self.nodes[node_idx];

        if node.hi - node.lo == 1 {
            return (Some(node.lo), None);
        }

        let l1 = node.rank.rank1(l);
        let r1 = node.rank.rank1(r);
        let l0 = l - l1;
        let r0 = r - r1;

        let left_non_empty = l0 < r0;
        let right_non_empty = l1 < r1;

        match (left_non_empty, right_non_empty) {
            (true, true) => {
                let (a, b) = self.range_bottom2_rec(node.left as usize, l0, r0);
                if b.is_some() {
                    (a, b)
                } else {
                    // Left had only 1 distinct value; second smallest is the right minimum.
                    let right_min = self.range_min_from(node.right as usize, l1, r1);
                    (a, right_min)
                }
            }
            (true, false) => self.range_bottom2_rec(node.left as usize, l0, r0),
            (false, true) => self.range_bottom2_rec(node.right as usize, l1, r1),
            (false, false) => (None, None),
        }
    }

    /// Minimum value in a subtree rooted at `node_idx` restricted to positions `[l, r)`.
    fn range_min_from(&self, mut node_idx: usize, mut l: usize, mut r: usize) -> Option<u32> {
        if l >= r {
            return None;
        }
        loop {
            let node = &self.nodes[node_idx];
            if node.hi - node.lo == 1 {
                return Some(node.lo);
            }
            let l1 = node.rank.rank1(l);
            let r1 = node.rank.rank1(r);
            let l0 = l - l1;
            let r0 = r - r1;
            if l0 < r0 {
                node_idx = node.left as usize;
                l = l0;
                r = r0;
            } else {
                node_idx = node.right as usize;
                l = l1;
                r = r1;
            }
        }
    }

    /// Push all distinct values present in `data[l..r)` into `buf` (in sorted order).
    pub fn range_distinct(&self, l: usize, r: usize, buf: &mut Vec<u32>) {
        if l >= r || r > self.n {
            return;
        }
        self.range_distinct_rec(0, l, r, buf);
    }

    fn range_distinct_rec(&self, node_idx: usize, l: usize, r: usize, buf: &mut Vec<u32>) {
        if l >= r {
            return;
        }

        let node = &self.nodes[node_idx];

        if node.hi - node.lo == 1 {
            buf.push(node.lo);
            return;
        }

        let l1 = node.rank.rank1(l);
        let r1 = node.rank.rank1(r);
        let l0 = l - l1;
        let r0 = r - r1;

        if l0 < r0 {
            self.range_distinct_rec(node.left as usize, l0, r0, buf);
        }
        if l1 < r1 {
            self.range_distinct_rec(node.right as usize, l1, r1, buf);
        }
    }
}

/*
impl<R: RankSupport, S: SelectSupport> sbwt::ContractLeft for WaveletTree<R,S> {
    fn contract_left(&self, I: std::ops::Range<usize>, target_len: usize) -> std::ops::Range<usize> {

        let new_start = match self.prev_smaller(I.start + 1, target_len as u32) {
            None => 0,
            Some(s) => s,
        };

        assert!(I.end > 0);
        let new_end = match self.next_smaller(I.end - 1, target_len as u32) {
            None => self.len(),
            Some(s) => s,
        };

        new_start..new_end
    }
}
    */

#[cfg(test)]
mod stress {

    use super::*;

    // --- Deterministic RNG (fast, no deps) ---

    #[derive(Clone)]
    struct SplitMix64(u64);
    impl SplitMix64 {
        fn new(seed: u64) -> Self {
            Self(seed)
        }
        fn next_u64(&mut self) -> u64 {
            // splitmix64
            self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            z ^ (z >> 31)
        }
        fn next_usize(&mut self) -> usize {
            (self.next_u64() >> 1) as usize
        }
        fn gen_range_u32(&mut self, lo: u32, hi: u32) -> u32 {
            assert!(lo < hi);
            let span = (hi - lo) as u64;
            lo + (self.next_u64() % span) as u32
        }
        fn gen_range_usize(&mut self, lo: usize, hi: usize) -> usize {
            assert!(lo < hi);
            let span = (hi - lo) as u64;
            lo + (self.next_u64() % span) as usize
        }
        fn coin(&mut self, num: u64, den: u64) -> bool {
            debug_assert!(num <= den && den > 0);
            (self.next_u64() % den) < num
        }
    }

    // --- Brute references ---

    fn brute_next_smaller(a: &[u32], i: usize, x: u32) -> Option<usize> {
        if i + 1 >= a.len() {
            return None;
        }
        for j in (i + 1)..a.len() {
            if a[j] < x {
                return Some(j);
            }
        }
        None
    }

    fn brute_prev_smaller(a: &[u32], i: usize, x: u32) -> Option<usize> {
        if i == 0 {
            return None;
        }
        for j in (0..i).rev() {
            if a[j] < x {
                return Some(j);
            }
        }
        None
    }

    // --- Data generators that are intentionally nasty for wavelet trees ---

    /// Generate data in [0, k) with a mixture of:
    /// - long runs (highly compressible)
    /// - alternating patterns (maximizes crossings between partitions)
    /// - heavy skew (most values near 0 or k-1)
    /// - occasional uniform noise
    fn gen_adversarial_array(rng: &mut SplitMix64, n: usize, k: u32) -> Vec<u32> {
        assert!(k >= 2);
        let mut a = Vec::with_capacity(n);
        let mut mode = 0u8;

        let mut i = 0usize;
        while i < n {
            // Switch mode sometimes
            if rng.coin(1, 20) {
                mode = (rng.next_u64() % 5) as u8;
            }

            match mode {
                // 0: long run of same value
                0 => {
                    let v = if rng.coin(1, 2) { 0 } else { k - 1 };
                    let run = rng.gen_range_usize(1, (n - i).min(512) + 1);
                    for _ in 0..run {
                        a.push(v);
                    }
                    i += run;
                }
                // 1: alternating low/high
                1 => {
                    let run = rng.gen_range_usize(1, (n - i).min(1024) + 1);
                    for t in 0..run {
                        a.push(if (t & 1) == 0 { 0 } else { k - 1 });
                    }
                    i += run;
                }
                // 2: staircase: 0,0,1,1,2,2,... wrapping
                2 => {
                    let run = rng.gen_range_usize(1, (n - i).min(1024) + 1);
                    let mut v = rng.gen_range_u32(0, k);
                    for t in 0..run {
                        // change every 2 steps
                        if (t % 2) == 0 {
                            v = (v + 1) % k;
                        }
                        a.push(v);
                    }
                    i += run;
                }
                // 3: heavy skew to small numbers, occasional spikes
                3 => {
                    let run = rng.gen_range_usize(1, (n - i).min(1024) + 1);
                    for _ in 0..run {
                        let r = rng.next_u64() % 100;
                        let v = if r < 85 {
                            // very small
                            (rng.next_u64() % 8) as u32 % k
                        } else if r < 95 {
                            // very large
                            k - 1 - ((rng.next_u64() % 8) as u32 % k)
                        } else {
                            // uniform
                            rng.gen_range_u32(0, k)
                        };
                        a.push(v);
                    }
                    i += run;
                }
                // 4: uniform noise
                _ => {
                    let run = rng.gen_range_usize(1, (n - i).min(2048) + 1);
                    for _ in 0..run {
                        a.push(rng.gen_range_u32(0, k));
                    }
                    i += run;
                }
            }
        }

        a
    }

    // --- Stress test ---

    /// A much more challenging stress test:
    /// - Larger n
    /// - Larger k (log k bigger)
    /// - Adversarial distributions
    /// - Many random queries, including edge and boundary x values
    /// - Some targeted x values around midpoints and around present values
    #[test]
    fn stress_next_prev_smaller_hard() {
        let mut rng = SplitMix64::new(0xD1CE_BA5E_F00D_F00Du64);

        // Keep this reasonably sized for unit tests; bump n/iters locally if desired.
        // If you want a "burn-in" stress test, set n=200_000, queries=200_000.
        let n: usize = 50_000;
        let k: u32 = 1 << 16; // 65536 alphabet -> log2(k)=16 levels

        let a = gen_adversarial_array(&mut rng, n, k);

        let wt = WaveletTree::new(
            &a, 0, k, 
            RankSupportV::<Pat1>::new, 
            |bv| SelectBinarySearchOverRank {
                rs : RankSupportV::<Pat1>::new(bv)
            }
        );

        // Query budget: mix of random and targeted
        let queries = 80_000usize;

        for t in 0..queries {
            // Choose i with bias toward edges and random interior
            let i = if t % 10 == 0 {
                // edge-heavy
                match t % 4 {
                    0 => 0,
                    1 => n / 2,
                    2 => n - 1,
                    _ => rng.gen_range_usize(0, n),
                }
            } else {
                rng.gen_range_usize(0, n)
            };

            // Choose x: include boundaries, midpoints, near-by values, and random
            let x = match t % 12 {
                0 => 0,
                1 => 1,
                2 => k - 1,
                3 => k,
                4 => k + 1, // outside range on purpose
                5 => k / 2,
                6 => (k / 2).saturating_sub(1),
                7 => (k / 2) + 1,
                8 => a[i].saturating_add(1), // just above current
                9 => a[i],                   // equal to current
                10 => a[i].saturating_sub(1), // just below current
                _ => rng.gen_range_u32(0, k),
            };

            let got_n = wt.next_smaller(i, x);
            let exp_n = brute_next_smaller(&a, i, x);
            assert_eq!(
                got_n, exp_n,
                "next_smaller mismatch @t={}: i={}, x={}, got={:?}, exp={:?}",
                t, i, x, got_n, exp_n
            );

            let got_p = wt.prev_smaller(i, x);
            let exp_p = brute_prev_smaller(&a, i, x);
            assert_eq!(
                got_p, exp_p,
                "prev_smaller mismatch @t={}: i={}, x={}, got={:?}, exp={:?}",
                t, i, x, got_p, exp_p
            );

            // Additional sanity: if an answer exists, verify it is actually < x and in correct direction
            if let Some(j) = got_n {
                assert!(j > i);
                assert!(a[j] < x);
                // ensure it's the next such position
                for jj in (i + 1)..j {
                    assert!(
                        !(a[jj] < x),
                        "next_smaller not minimal: found earlier jj={} with a[jj]={} < x={}",
                        jj,
                        a[jj],
                        x
                    );
                }
            }
            if let Some(j) = got_p {
                assert!(j < i);
                assert!(a[j] < x);
                for jj in (j + 1)..i {
                    assert!(
                        !(a[jj] < x),
                        "prev_smaller not maximal: found later jj={} with a[jj]={} < x={}",
                        jj,
                        a[jj],
                        x
                    );
                }
            }
        }
    }

    #[test]
    fn stress_range_distinct() {
        let mut rng = SplitMix64::new(0xCAFE_BABE_DEAD_BEEFu64);

        let n: usize = 10_000;
        let k: u32 = 256;

        let a = gen_adversarial_array(&mut rng, n, k);

        let wt = WaveletTree::new(
            &a, 0, k,
            RankSupportV::<Pat1>::new,
            |bv| SelectBinarySearchOverRank {
                rs: RankSupportV::<Pat1>::new(bv),
            },
        );

        let queries = 20_000usize;
        let mut buf = Vec::new();

        for _ in 0..queries {
            let l = rng.gen_range_usize(0, n);
            let r = rng.gen_range_usize(l, n + 1);

            buf.clear();
            wt.range_distinct(l, r, &mut buf);

            // Brute force: collect distinct values from a[l..r), sorted
            let mut expected: Vec<u32> = a[l..r].iter().copied().collect();
            expected.sort_unstable();
            expected.dedup();

            assert_eq!(
                buf, expected,
                "range_distinct mismatch: l={}, r={}, got={:?}, exp={:?}",
                l, r, buf, expected,
            );
        }
    }

    #[test]
    fn stress_range_bottom2() {
        let mut rng = SplitMix64::new(0x1234_5678_9ABC_DEF0u64);

        let n: usize = 10_000;
        let k: u32 = 256;

        let a = gen_adversarial_array(&mut rng, n, k);

        let wt = WaveletTree::new(
            &a, 0, k,
            RankSupportV::<Pat1>::new,
            |bv| SelectBinarySearchOverRank {
                rs: RankSupportV::<Pat1>::new(bv),
            },
        );

        let queries = 20_000usize;

        for _ in 0..queries {
            let l = rng.gen_range_usize(0, n);
            let r = rng.gen_range_usize(l, n + 1);

            let (got0, got1) = wt.range_bottom2(l, r);

            // Brute force: sorted distinct values
            let mut vals: Vec<u32> = a[l..r].iter().copied().collect();
            vals.sort_unstable();
            vals.dedup();

            let exp0 = vals.first().copied();
            let exp1 = if vals.len() > 1 { Some(vals[1]) } else { None };

            assert_eq!(
                (got0, got1), (exp0, exp1),
                "range_bottom2 mismatch: l={}, r={}, got=({:?},{:?}), exp=({:?},{:?})",
                l, r, got0, got1, exp0, exp1,
            );
        }
    }
}

