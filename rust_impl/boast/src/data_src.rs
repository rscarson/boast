use crate::Error;

type BoxedTransform<T> = Box<dyn Fn(&mut [T], u64)>;
type BoxedIter<T> = Box<dyn Iterator<Item = Vec<T>>>;

/// Source of data for a BOAST run, either transformed from an original dataset
/// or generated from scratch
pub enum DataSource<T: Clone> {
    /// Original dataset with a transformation function to produce test datasets
    ///
    /// The length of the transformed datasets MUST be the same as the original dataset
    Transformed {
        original: Vec<T>,
        transform: BoxedTransform<T>,
    },

    /// An iterable data source that produces datasets from a provided iterator
    /// The length of the datasets produced MUST be consistent across calls
    ///
    /// If the iterator is exhausted, the test will panic
    Iterable { length: usize, iter: BoxedIter<T> },
}
impl<T: Clone> DataSource<T> {
    /// Create a transformed data source from an original dataset and a transformation function
    ///
    /// The transformation function takes in a mutable slice of the original data to transform, and a seed for any necessary randomization.
    pub fn transformed<F>(original: Vec<T>, transform: F) -> Self
    where
        F: Fn(&mut [T], u64) + 'static,
    {
        Self::Transformed {
            original,
            transform: Box::new(transform),
        }
    }

    /// Create a generated data source from a generator function and the length of datasets to produce
    ///
    /// The generator function takes in a mutable slice of memory to write the generated data into, and a seed for any necessary randomization.
    pub fn generated<F>(length: usize, generator: F) -> Self
    where
        F: Fn(&mut [T], u64) + 'static,
        T: Default,
    {
        let container = vec![T::default(); length];
        Self::Transformed {
            original: container,
            transform: Box::new(generator),
        }
    }

    /// Create an iterable data source from a provided iterator
    pub fn iterable<I>(length: usize, iter: I) -> Self
    where
        I: Iterator<Item = Vec<T>> + 'static,
    {
        Self::Iterable {
            length,
            iter: Box::new(iter),
        }
    }

    /// Get the next dataset to test, using the provided seed for any necessary transformations
    /// or generation.
    ///
    /// Can return an error if the data source is exhausted
    pub fn get_data(&mut self, seed: u64) -> Result<Vec<T>, Error> {
        match self {
            DataSource::Transformed {
                original,
                transform,
            } => {
                let mut data = original.clone();
                transform(&mut data, seed);
                Ok(data)
            }
            DataSource::Iterable { iter, .. } => iter.next().ok_or(Error::DataSourceExhausted),
        }
    }

    /// Get the length of the datasets produced by this data source
    pub fn len(&self) -> usize {
        match self {
            DataSource::Transformed { original, .. } => original.len(),
            DataSource::Iterable { length, .. } => *length,
        }
    }

    /// Check if the data source produces empty datasets
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/* From a countable iterator */
impl<T: Clone, I> From<I> for DataSource<T>
where
    I: ExactSizeIterator<Item = Vec<T>> + 'static,
{
    fn from(value: I) -> Self {
        Self::iterable(value.len(), value)
    }
}
