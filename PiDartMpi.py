from __future__ import division, print_function
import math
import random
from mpi4py import MPI

import time # optional, for timing only

def circle_throws(throws):
  i = 0
  hits = 0
  while i < throws:
    x = random.random()
    y = random.random()
    if (x * x + y * y) < 1.:
      hits = hits + 1
    i = i + 1
  return hits

if __name__ == '__main__':
  startTime = time.time()

  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()

  throws = 1e8 // size
  loops = 1e4

  throws_per_loop = throws // loops
  throwsAllProcs = throws_per_loop * loops * size

  rootProc = 0

  i = 0
  hitsAllProcs = 0
  while i < loops:
    hits = circle_throws(throws_per_loop)
    red = comm.reduce(hits, op=MPI.SUM, root=rootProc)
    if rank == rootProc:
      hitsAllProcs = hitsAllProcs + red
    i = i + 1

  if rank == rootProc:
    pi = 4.0 * float(hitsAllProcs) / float(throwsAllProcs)

    piError = abs(pi - math.pi)/math.pi
    print("Pi estimate: "+str(pi))
    print("Pi error: "+str(piError))
    print("Darts thrown: "+str(throwsAllProcs))
    print("Runtime (s): "+str(time.time()-startTime))
