#  x07.txt
#
#  Reference:
#
#    D G Kleinbaum and L L Kupper,
#    Applied Regression Analysis and Other Multivariable Methods,
#    Duxbury Press, 1978, page 148.
#
#    Helmut Spaeth,
#    Mathematical Algorithms for Linear Regression,
#    Academic Press, 1991,
#    ISBN 0-12-656460-4.
#
#  Discussion:
#
#    A psychiatrist assigns values for mental retardation and 
#    degree of distrust of doctors in 53 newly hospitalized patients.
#    Six months later, a value for the degree of illness of each patient
#    is assigned.  The question is whether the effect of treatment can
#    be related to these indices.  Since the independent and dependent
#    variables are essentially subjective, it is assumed that large
#    errors are made in all of them.
#
#    There are 53 rows of data.  The data comprises:
#
#      I,  the index;
#      A1, the degree of mental retardation;
#      A2, the degree of distrust of doctors;
#      B,  the degree of illness.
#
#    We seek a model of the form:
#
#      B = A1 * X1 + A2 * X2
#
4 columns
53 rows
Index
Retardation Index
Distrust Index
Degree of Illness after 6 Months
 1  2.80  6.1  44
 2  3.10  5.1  25
 3  2.59  6.0  10
 4  3.36  6.9  28
 5  2.80  7.0  25
 6  3.35  5.6  72
 7  2.99  6.3  45
 8  2.99  7.2  25
 9  2.92  6.9  12
10  3.23  6.5  24
11  3.37  6.8  46
12  2.72  6.6   8
13  3.47  8.4  15
14  2.70  5.9  28
15  3.24  6.0  26
16  2.65  6.0  27
17  3.41  7.6   4
18  2.58  6.2  14
19  2.81  6.0  21
20  2.80  6.4  22
21  3.62  6.8  60
22  2.74  8.4  10
23  3.27  6.7  60
24  3.78  8.3  12
25  2.90  5.6  28
26  3.70  7.3  39
27  3.40  7.0  14
28  2.63  6.9   8
29  2.65  5.8  11
30  3.26  7.2   7
31  3.15  6.5  23
32  2.60  6.3  16
33  2.74  6.8  26
34  2.72  5.9   8
35  3.11  6.8  11
36  2.79  6.7  12
37  2.90  6.7  50
38  2.74  5.5   9
39  2.70  6.9  13
40  3.08  6.3  22
41  2.18  6.1  23
42  2.88  5.8  31
43  3.04  6.8  20
44  3.32  7.3  66
45  2.80  5.9   9
46  3.29  6.8  12
47  3.56  8.8  21
48  2.74  7.1  13
49  3.06  6.9  10
50  2.54  6.7   4
51  2.78  7.2  18
52  2.81  5.2  10
53  3.26  6.6   7