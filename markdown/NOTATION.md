# Mathematical notation and output conventions

This notation describes the implemented interfaces, not new mathematical claims.
Full construction contracts are in [CONTRACTS.md](CONTRACTS.md).
Version 0.3 extensions and executable API conventions are in
[CONTRACTS_0_3.md](CONTRACTS_0_3.md).

* A point-cloud record is `(X, I, w, metadata)`: finite coordinates, unique stable
  IDs, assigned scalar weights, and scientific attributes. File format is not
  part of a mathematical core's input contract. Molecular bonds, partial/formal
  charges, atom names/types, unit cells, and raw source records remain metadata
  until an application explicitly selects how to use them.
* A builder produces a mathematical object or scalar filtration `K(t)`, `D(t)`,
  or `IC(t)` for a simplicial, sequence-hyperdigraph, or two-factor interaction
  route. Their chain spaces are distinct and are not implicitly interchangeable.
* `C_q(t)` denotes the relevant degree-q chain space, with boundary `B_q(t)`.
  In any valid chain complex, `B_q B_(q+1) = 0`. Embedded and interaction chain
  bases can be combinations/quotients; they are not necessarily individual cells.
* `H_q` is homology over the specified field. Persistent intervals are
  `[birth, death)` in that field; infinite death means survival in the supplied
  filtration, possibly cutoff-censored.
* In an orthonormal real chain basis, ordinary
  `L_q(t) = B_q(t)^T B_q(t) + B_(q+1)(t) B_(q+1)(t)^T`.
  Non-coordinate embedded bases require the corresponding induced metric.
  Eigenvectors are labelled in the returned basis, not assumed to be point vectors.
* A genuine persistent `L_q(s,t)` acts on the source-time basis with the
  appropriate restricted target boundary space. It is not `L_q(t)-L_q(s)` and
  is not a list of independent ordinary snapshots. Both operations are exposed
  separately; snapshots are the default.
* Full Lq and correct Hq deaths require q+1 boundary information. Explicit
  construction caps define truncated objects and can change results.

For ordinary alpha, filtration is squared coordinate radius; assigned weights
are ignored. Default hyperdigraph support is Delaunay adjacency (explicit bonds
and a distance cutoff may replace/restrict it); low-to-high weights
orient edges, exact ties retain both directions. Sequence births use maximum
consecutive-edge distance. Interaction has two independently built factors
and a scalar maximum coupling with explicit shared IDs. These different
scale conventions cannot be identified merely because their numbers coincide.
For an explicitly paired schedule, `IC(j) = IC(K_A(a_j), K_B(b_j))`, where
`a_j` and `b_j` are nondecreasing. Grades `j` or a supplied strictly increasing
progression label the sampled one-parameter path. Ordinary pair snapshots and
genuine inclusion operators `(a_s,b_s) -> (a_t,b_t)` remain distinct.

## Numerical postprocessing

Barcode features bin numerical births/deaths, with separate essential-interval
and overflow channels. Raw results are not edited by feature thresholds.

The separate `postprocessing.barcode_bin_counts` API returns one count per
explicit one-dimensional bin per selected homology degree. For a positive-
length bar `[b,d)` and bin `[l,r)`, `mode="overlap"` counts it when `b < r`
and `d > l`; `mode="cover"` counts it when `b <= l` and `d >= r`. Thus a
bar dying exactly at `r` covers `[l,r)`, while a bar born exactly at `r`
does not overlap that bin. Zero-length bars never contribute. Neither mode
is a sampled Betti curve: short bars within a bin contribute to overlap
even if absent at both endpoints.

`mode="death"` counts finite deaths in left-closed/right-open bins, including
the final right endpoint in the last bin. Infinite-death bars contribute to
overlap and coverage but not death histograms. Infinite deaths are never
replaced by the bin maximum. Bins must stay within any declared filtration
window; survivors at a finite cutoff are not asserted to survive beyond it.
No extra overflow columns, normalization, endpoint rounding or learned grid
are added. Dimension order then bin order defines the flattened vector, and
the full rule and original filtration units remain in its schema/provenance.
For example, boundaries `0, 0.05, ..., 5` define 100 bins, not 101 features.
Alpha builders report squared-radius births; any application choosing radius
or diameter units must transform barcode endpoints and their metadata before
calling this numerical postprocessing API.

Spectral summaries operate only on supplied eigenvalues. Mean/min/max/std use
positive eigenvalues by default; `positive_only=False` selects all values. A
separate zero count uses the full spectrum and is undefined for partial spectra.
Empty undefined
statistics are NaN, not zero. Partial spectra require explicit opt-in and never
become a claim about the full spectral distribution.

If `dim C_q(s) = 0`, the ordinary or persistent degree-q operator on that
source basis is the complete `0 x 0` operator. Its spectrum is empty and its
nullity is zero; inserting the artificial spectrum `[0]` would instead claim a
one-dimensional kernel. For fixed-width feature encoding only,
`empty_operator_policy="zero"` may map every requested predefined scalar
summary of this complete empty operator to zero after the basis, nullity,
optional matrices, and standardized core metadata jointly confirm structural
absence. Custom callbacks retain their own empty-input definitions. The raw
spectrum stays empty, and the policy does not apply when `dim C_q(s) > 0` but
all eigenvalues are zero, when only the positive-mode selection is empty, or
when a spectrum is partial.

For a selected spectrum of size `n > 0`, the raw order-`k` spectral moment is
`m_k = (1/n) sum_i lambda_i^k`, and spectral variance is the population central
second moment `(1/n) sum_i (lambda_i - m_1)^2`. The named `moment_1` through
`moment_4` summaries use the same current selection as mean unless
`positive_only=False`. Spectral energy is `sum_i |lambda_i|`. The distinct
centered Laplacian spectral energy is
`LE = sum_i |lambda_i - mean(lambda)|`; it always uses the complete full
operator spectrum, including numerical zero modes, and is unavailable for a
partial spectrum. This equals classical graph Laplacian energy for a
combinatorial graph L0, where `mean(lambda) = 2m/n`. For Hyperdigraph L0/L1 or
other operators it is a declared generalization rather than a standard graph
invariant.

For a positive-semidefinite spectrum, define `p_i = lambda_i / sum(lambda)`
over positive entries and `S = -sum(p_i log(p_i))`. With graph L0, this is
trace-normalized spectral/von Neumann entropy, not degree-distribution entropy.
Natural log is the default; the zero operator has entropy zero by documented
convention. Repeated eigenvectors are nonunique; compare invariant subspaces
when needed. Real spectral nullity is not universally the GF(2) Betti number.

## Visual notation

A displayed q-simplex has q+1 vertices. Triangular faces are filled; a
tetrahedron displays four faces. Higher-dimensional simplices display a
labelled 2-skeleton, not their interior. Showing the edges of a graph does not
fill cliques automatically.

An ordered q-hyperedge `(v0, ..., vq)` is represented by q consecutive arrows
`v0 → ... → vq`, with ribbon color indicating degree. Arrows and endpoint
markers describe this sequence; they do not assert additional hyperedges,
singleton membership, or an embedded-chain basis. Curved lanes for reciprocal
or repeated support are display offsets, not altered coordinates. Canonical
building-block layouts are illustrative. Full conventions and limitations are
in [VISUALIZATION.md](VISUALIZATION.md).
