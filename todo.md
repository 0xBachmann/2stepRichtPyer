# TODO's
- [x] initial values better
- [x] 2D
- [x] waves for 2D
  - not possible
  - [x] rotate 1d waves
    - [x] now check accuracy
    - [x] check correctness of rotating and a sound depending on w0 ofc wrong
- [ ] derivative maybe not correct and add for burgers
- [ ] TODO's in src
- [ ] implicit scheme
  - [x] speed not correct...
    - correct when using avg = id
  - [x] fails for dt too large
    - [x] check jacobian
      - correct now
    - [x] works now without jacobian
    - works for root but not newton
      - [x] try giving J to root
        - now do also for 1d
        - but not so fast
  - [x] for linear advection
    - [x] linear stability analysis
- [x] vortex test
  - works now when using arctan2 correctly
  - [ ] check energy conservation
- [ ] reference solutions?
- [ ] review CFL for diagonal motion
- [x] neumann stability analysis
  - done in 1D
- [x] check triangles in accuracy plotter for correctness
- [x] plotter dimensions correct order
- [x] submit short description
- [ ] dirichlet bd
- [x] does implicit really represent two-step?
  - yes
- [ ] refactor `PDE.derivative`
- [ ] lsa 2d try in state space
- [x] sound wave through vortex
  - works perfectly
  - but $M_r$ ??
- [ ] Kelvin Helmholtz
- [ ] writeout only binary vals
- [x] `fig.tight_layout()`
  - maybe more occurrences needed?
- [ ] add scalar advection euler


# To Discuss
## 06.04.23
- bachelor thesis proposal
- convergence of rot wave, only n * 45°
- gresho vortex problem
  - cfl?
  - init?
  - evolution?
  - bdc?
- implicit not stable for dt > 2 * dt_crit
## 21.04.23
- implicit works but not with newton
## 28.04.23
- reference Mach $M_r$