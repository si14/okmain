#[derive(Debug, Copy, Clone, PartialEq)]
pub struct Oklab {
    pub l: f32,
    pub a: f32,
    pub b: f32,
}

impl Oklab {
    pub(crate) fn squared_distance(self, other: Self) -> f32 {
        let dl = self.l - other.l;
        let da = self.a - other.a;
        let db = self.b - other.b;
        dl.mul_add(dl, da.mul_add(da, db * db))
    }
}
