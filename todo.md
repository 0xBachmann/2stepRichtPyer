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
  - [ ] fails for dt too large
    - [ ] check jacobian
    - [x] works now without jacobian
  - [x] for linear advection
    - [x] linear stability analysis
- [ ] vortex test
- [ ] reference solutions?
- [ ] review CFL for diagonal motion
- [ ] neumann stability analysis
- [x] check triangles in accuracy plotter for correctness
- [x] plotter dimensions correct order
- [x] submit short description
- [ ] dirichlet bd


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