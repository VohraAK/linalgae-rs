use std::fmt::{self, Display};
use std::ops::Fn;
use num_traits::Num;

// defining a matrix struct (generic type)
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix<T>
where T : Num
{
    rows: usize,
    cols: usize,
    data: Vec<T>
}

impl<T> Matrix<T>
where T: Num
{
    // "default" constructor
    pub fn new(rows: usize, cols: usize, data: Vec<T>) -> Result<Self, &'static str>
    {
        // check rows and cols
        if rows <= 0 { return Err("Row dim should be positive!"); }

        if cols <= 0 { return Err("Col dim should be positive!"); }

        // check if number of elements equal to (rows x cols)
        if (rows * cols) != data.len() { return Err("Input vector does not match dimensions!"); }
        
        Ok(Matrix { rows, cols, data })
    }

    // "full" constructor, returns a Matrix filled with <fill_value>
    pub fn full(rows: usize, cols: usize, fill_value: T) -> Result<Self, &'static str>
    where T: Clone
    {
        // check rows and cols
        if rows <= 0 { return Err("Row dim should be positive!"); }

        if cols <= 0 { return Err("Col dim should be positive!"); }

        Ok(Matrix { rows, cols, data: vec![fill_value; rows * cols] })
    }

    // "zeros" constructor, returns a zeroed out Matrix
    pub fn zeroes(rows: usize, cols: usize) -> Result<Self, &'static str> 
    where T: Num + Clone + Copy
    {
        Self::full(rows, cols, T::zero())
    }

    // "ones" constructor, returns a Matrix filled with ones
    pub fn ones(rows: usize, cols: usize) -> Result<Self, &'static str> 
    where T: Clone + num_traits::One
    {
        Self::full(rows, cols, T::one()) 
    }

    // "identity" constructor, returns an identity Matrix
    pub fn identity(n: usize) -> Result<Self, &'static str>
    where T: Num + Clone + Copy
    {
        if n <= 0 { return Err("Dimension should be positive!"); }

        let mut mat = Self::zeroes(n, n)?;
        let data = mat.as_mut_slice();

        for i in 0..n
        {
            data[i*n + i] = T::one();
        }

        Ok(mat)
    }

    // dim getters
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    // data accessors implemented as slicers
    pub fn as_slice(&self) -> &[T] { &self.data }
    pub fn as_mut_slice(&mut self) -> &mut[T] { &mut self.data }

    // transpose function
    pub fn t(&self) -> Matrix<T>
    where T: Default + Clone + Copy,
    {
        let rows = self.rows();
        let cols = self.cols();
        let mut result = vec![T::zero(); rows * cols];
        let slice = self.as_slice();

        for i in 0..rows 
        { 
            for j in 0..cols 
            { 
                result[j*rows +i] = slice[i*cols + j]; 
            } 
        }

        Matrix::new(cols, rows, result).unwrap()

    }

    // apply function -> applies a closure / function to all values of the matrix
    // using Fn syntax since it can both capture variables from outer scopes, and act as a regular function
    pub fn apply(&self, func: impl Fn(T) -> T) -> Matrix<T>
    where T: Copy
    {
        let result = self.data.as_slice().iter().map(|x| func(*x)).collect();

        Matrix::new(self.rows, self.cols, result).unwrap()
    }

}

// X----------X
// impl fmt::Display trait for Matrix
impl<T> fmt::Display for Matrix<T>
where T: Num + Display
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result 
    {
        write!(f, "\n[")?;

        for row in 0..self.rows
        {
            write!(f, "[")?;

            for col in 0..self.cols
            {
                let idx = row * self.cols + col;
                write!(f, "{:.2}", self.data[idx])?;

                if col != self.cols - 1
                {
                    write!(f, ", ")?;
                }
            }

            write!(f, "]")?;

            if row != self.rows - 1
            {
                write!(f, ",\n")?;
            }
        }

        write!(f, "]")?;

        Ok(())
    }
}