use std::fmt;
use std::ops::Fn;

// defining a matrix struct (f64)
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix
{
    rows: usize,
    cols: usize,
    data: Vec<f64>
}

impl Matrix
{
    // "default" constructor
    pub fn new(rows: usize, cols: usize, data: Vec<f64>) -> Result<Self, &'static str>
    {
        // check rows and cols
        if rows <= 0 { return Err("Row dim should be positive!"); }

        if cols <= 0 { return Err("Col dim should be positive!"); }

        // check if number of elements equal to (rows x cols)
        if (rows * cols) != data.len() { return Err("Input vector does not match dimensions!"); }
        
        Ok(Matrix { rows, cols, data })
    }

    // "full" constructor, returns a Matrix filled with <fill_value>
    pub fn full(rows: usize, cols: usize, fill_value: f64) -> Result<Self, &'static str>
    {
        // check rows and cols
        if rows <= 0 { return Err("Row dim should be positive!"); }

        if cols <= 0 { return Err("Col dim should be positive!"); }

        Ok(Matrix { rows, cols, data: vec![fill_value as f64; rows * cols] })
    }

    // "zeros" constructor, returns a zeroed out Matrix
    pub fn zeroes(rows: usize, cols: usize) -> Result<Self, &'static str> { Self::full(rows, cols, 0 as f64) }

    // "ones" constructor, returns a Matrix filled with ones
    pub fn ones(rows: usize, cols: usize) -> Result<Self, &'static str> { Self::full(rows, cols, 1 as f64) }

    // dim getters
    pub fn rows(&self) -> usize { self.rows }
    pub fn cols(&self) -> usize { self.cols }

    // data accessors implemented as slicers
    pub fn as_slice(&self) -> &[f64] { &self.data }
    pub fn as_mut_slice(&mut self) -> &mut[f64] { &mut self.data }

    // transpose function
    pub fn t(&self) -> Matrix
    {
        let rows = self.rows();
        let cols = self.cols();
        let mut result = vec![0 as f64; rows * cols];
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
    pub fn apply(&self, func: impl Fn(f64) -> f64) -> Matrix
    {
        let result = self.data.as_slice().iter().map(|x| func(*x)).collect();

        Matrix::new(self.rows, self.cols, result).unwrap()
    }

}

// X----------X
// impl fmt::Display trait for Matrix
impl fmt::Display for Matrix
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