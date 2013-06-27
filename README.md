# Test of liblinear

A test of the machine learning library clj-liblinear.

Datasets taken from http://www.cs.cornell.edu/people/pabo/movie-review-data/

Tested one model for subjectivity/objectivity. That worked well. Test another model for negative/positive sentiment. Did not work so well.

Results:

```
subjective/objective test (0 is objective, 1 is subjective)
{:objective {0.0 1132, 1.0 118}, :subjective {1.0 1103, 0.0 147}}
negative/positive test (0 is negative, 1 is positive)
{:negative {1.0 2874, 0.0 5072}, :positive {1.0 5106, 0.0 3128, #<ClassCastException java.lang.ClassCastException: [Ljava.lang.Object; cannot be cast to [Lde.bwaldvogel.liblinear.Feature;> 1}}

```
