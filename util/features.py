from math import sqrt
from util.const import BINS

def _avg(vals):
    N = len(vals)
    sumx = 0
    sumy = 0
    sumz = 0
    
    for val in vals:
        sumx += val[0]
        sumy += val[1]
        sumz += val[2]

    return (sumx / N, sumy / N, sumz / N)

def _stddev(vals, means):
    N = len(vals)
    sumx = 0
    sumy = 0
    sumz = 0

    for val in vals:
        sumx += (val[0] - means[0]) ** 2
        sumy += (val[1] - means[1]) ** 2
        sumz += (val[2] - means[2]) ** 2

    return [sqrt(sumx / (N - 1)), sqrt(sumy / (N - 1)), sqrt(sumz / (N - 1))]

def _absdev(vals, means):
    N = len(vals)
    sumx = 0
    sumy = 0
    sumz = 0

    for val in vals:
        sumx += abs(val[0] - means[0])
        sumy += abs(val[1] - means[1])
        sumz += abs(val[2] - means[2])

    return [sumx / N, sumy / N, sumz / N]

def _resultant(vals):
    N = len(vals)
    summ = 0

    for val in vals:
        summ += sqrt(val[0] ** 2 + val[1] ** 2 + val[2] ** 2)

    return summ / N

def _zerocross(vals):
    N = len(vals)
    countx = 0
    county = 0
    countz = 0
    
    for i in range(1, len(vals)):
        if vals[i][0]*vals[i-1][0] < 0:
            countx += 1
        
        if vals[i][1]*vals[i-1][1] < 0:
            county += 1
            
        if vals[i][2]*vals[i-1][2] < 0:
            countz += 1

    return [float(countx) / N, float(county) / N, float(countz) / N]

def _minima(vals):
    minx = vals[0][0]
    miny = vals[0][1]
    minz = vals[0][2]

    for val in vals:
        if minx > val[0]:
            minx = val[0]

        if miny > val[1]:
            miny = val[1]

        if minz > val[2]:
            minz = val[2]

    return [minx, miny, minz]

def _maxima(vals):
    maxx = vals[0][0]
    maxy = vals[0][1]
    maxz = vals[0][2]

    for val in vals:
        if maxx < val[0]:
            maxx = val[0]

        if maxy < val[1]:
            maxy = val[1]

        if maxz < val[2]:
            maxz = val[2]

    return [maxx, maxy, maxz]

def _hist(vals, minval, maxval):
    N = len(vals)
    binx = [0] * BINS
    biny = [0] * BINS
    binz = [0] * BINS
    step = (maxval - minval) / float(BINS)

    for val in vals:
        if val[0] < minval:
            binx[0] += 1
        elif val[0] > maxval:
            binx[BINS - 1] += 1
        else:
            for i in range(BINS-1, -1, -1):
                if val[0] > (minval + i * step):
                    binx[i] += 1
                    break

    for val in vals:
        if val[1] < minval:
            biny[0] += 1
        elif val[1] > maxval:
            biny[BINS - 1] += 1
        else:
            for i in range(BINS-1, -1, -1):
                if val[1] > minval + i * step:
                    biny[i] += 1
                    break

    for val in vals:
        if val[2] < minval:
            binz[0] += 1
        elif val[2] > maxval:
            binz[BINS - 1] += 1
        else:
            for i in range(BINS-1, -1, -1):
                if val[2] > minval + i * step:
                    binz[i] += 1
                    break
    
    for i in range(BINS):
        binx[i] /= float(N)
        biny[i] /= float(N)
        binz[i] /= float(N)

    res = []
    res.extend(binx)
    res.extend(biny)
    res.extend(binz)
    return res
