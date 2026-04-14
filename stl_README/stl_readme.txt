=================
README for stl_robustness.py
@author Kasidit Muenprasitivej
=================

Classes and How they compute robustness
---------
Robustness is computed by calling .robustness(t, x) on the root node.
Each node asks its children to compute first, then combines their results.
This repeats recursively until reaching a Predicate, which returns raw values.


STLNode
└── Predicate   : converts string "x[0] - 1.0" into numpy evalution given value of x
└── Negation    : require one child to define this class, and the robustness is just the negative of its child's robustness
└── Conjunction : require two childrens (left and right) to define this class, and then take min of the two child's robustness
└── Disjunction : require two childrens (left and right) to define this class, and then take max of the two child's robustness
|
└── TemporalOperator : This extended class add base function _window_indices() --> Given current time t[i] and interval [a,b], find all indices j such that t[i] + a < t[j] < t[i] + b
    └── Globally     : compute robustness of its child over the entire trajectory, and then take min over the time interval given by _window_indices()
    └── Eventually   : compute robustness of its child over the entire trajectory, and then take max over the time interval given by _window_indices()


=================
How to write STL formula into yaml format
=================

YAML
-----------
Every STL node must have a 'type' key.  Additional keys depend on type:

  type: predicate
    expression: "<Python expression using x[i] and np>"
    
    NOTE:   Here x[i] refer to the dimension of the state during numpy evaluation of the string
            so "x[0]" means the state must be atleast 1D

  type: NOT
    child: <STLNode>

  type: AND | OR
    left:  <STLNode>
    right: <STLNode>

  type: G | F
    interval: [a, b]   # time values
    child: <STLNode>


=================
FULL TRACE EXAMPLE:  G[0,10] F[0,5] (x[0] > 1.0  AND  x[0] < 2.5)
=================

YAML file structure:
see "stl_nested_spec_example.yaml" on how to define nested STL formula


1. Create STL formula in the code

  Tree structure:
    Globally
    └── Eventually
        └── Conjunction
            └── Predicate("x[0] - 1.0")
            └── Predicate("2.5 - x[0]")

  Step 1 — Globally calls Eventually
  Step 2 — Eventually calls Conjunction
  Step 3 — Conjunction calls both Predicates
  Step 4 — Predicates evaluate and return raw arrays      (bottom of recursion)

          Predicate("x[0] - 1.0")  --> [0.5,  1.2, -0.3, ...]
          Predicate("2.5 - x[0]")  -->  [1.0,  0.8,  1.7, ...]

2. Compute Robustness

  Step 5 — Conjunction takes elementwise min
          [0.5, 0.8, -0.3, ...]

  Step 6 — Eventually takes max over each [t+0, t+5] window
          [0.8, 0.8,  0.5, ...]

  Step 7 — Globally takes min over each [t+0, t+10] window
          [0.5, 0.5,  0.5, ...]           (final robustness signal)

