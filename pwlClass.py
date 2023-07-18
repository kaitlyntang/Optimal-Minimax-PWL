# -*- coding: utf-8 -*-

"""
Created on Monday July 17

@author: Kaitlyn Tang
"""

import numpy as np
import matplotlib.pyplot as plt
float_formatter = "{:.6f}".format
np.set_printoptions(formatter={'float_kind': float_formatter})
plt.rcParams['figure.figsize'] = [12, 10]


class pwl(object):
    """
    rev 0.2
    4/27/23

    A class to represent piecewise linear functions that approximate a convex or concave function over a given domain
    Requires numpy
    ...
    Attributes
    __________
    f : function
        The function that is being approximated by the PWL
    xmin : double
        The lower bound on the function domain
    xmax : double
        The upper bound on the function domain
    N : int
        The number of segments to be used by the PWL
    mode :
        The error metric being optimized: "ey" (vertical), "ex" (horizontal) or "en" (normal)
    loopCntlDict :
        A dictionary containing the recursion loop Control parameters
        for definitions, see method loopCntlSetup()

    pivot[] : N dimensional double np.array containing the pivots xcoord computed by fitPWL()
    knot[]  : N+1 dimensional double np.array containing the knots xcoord computed by fitPWL()
    fknot[] : N+1 dimensional double np.array containing the knots ycoord computed by fitPWL()
    slope[] : N dimensional double np.array containing the PWL slopes computed by fitPWL()
    err[]   : N+1 dimensional double np.array containing the PWL segment error at the knots
    error   : double containing the max PWL error evaluated at the knots

    Methods
    _______
    __init__() :
        Constructor pwl(f, xmin, xmax, N, mode(optional), loopCntlDict(optional))
        runs fitPWL() method to generate PWL functions
    fpwl(x) :
        Returns the PWL approximation of f() evaluated at x
    finvpwl(y):
        Returns the PWL approximation of finv() evaluated at y
    f():
        Evaluates the function f(x)
    df():
        Numerically evaluates the derivative f'(x) using f(x +/- delta)
    fitPWL():
        Finds pivots and knots to fit the PWL to the function using a recursive gradient descent approach
    loopCntlSetup():
        Sets loop control parameters to default if empty
        Adds or modifies control parameters passed as dictionary items

    Returns
    _______
        None

    Non-member functions
    _______
    Knot Initialization Functions to aid convergence
    knot_rescale():
        Scales knots into [xmin,xmax] range
    knot_double():
        Combine N+1 knots to 2N/2N+1/2N+2 knots
    knot_errscale():
        Scales knots into [xmin,xmax] range with spacing inversely proportional to errors

    Display Functions
    pwlfDisplay():
        Display the PWL function
    pwlfErrorDisplay():
        Compare the PWL function error relative to the target function
    pwlfinvDisplay():
        Display the inverse PWL function
    pwlfinvErrorDisplay():
        Compare the PWL function error relative to the target function
    pwlfTangentDisplay():
        Plot the PWL function tangents at pivots
    pwlKnotDetailDisplay():
        Display the PWL function Detail around a knot
    pwlPivotDisplay():
        Plot Knots and Pivots vs indices
    pwlPivotAngleDisplay():
        plot knot locations as angle in polar coordinates
        Interesting case when f(x) = sqrt(1 - x^2), because pivots should be linearly spaced

    """

    def __init__(self, fun, xmin, xmax, N=8, mode="ey", knot0=[], pivot0=[], loopCntlDict = {}):
        self.f = fun
        self.xmin = xmin
        if (xmax > xmin) :
            self.xmax = xmax
        else:
            print ("Can't __init__()")
            self.xmax = xmin
        if (0 <= N <= 1025):
            self.N = N
        else:
            print ("Can't __init__()")
            self.N = 0
        if (mode.lower() == "ex" or mode.lower() == "x") :
            self.mode = "ex"
        elif (mode.lower() == "ey" or mode.lower() == "y") :
            self.mode = "ey"
        elif (mode.lower() == "en" or mode.lower() == "n") :
            self.mode = "en"
        else:
            print ("Can't __init__()")
        self.loopCntl = {}
        self.loopCntlSetup(loopCntlDict) # sets default loop Control attributes
        self.loopCntlSetup({"dfDelta": (0.01 * self.loopCntl["relLim"] * (self.xmax - self.xmin) / self.N) })
        self.fitPWL(knot0, pivot0)

    def loopCntlSetup(self, loopCntlDict={}):
        """
        Sets loop Control parameters as included in dictionary
        set default values if empty

        Parameters
        ----------
        loopCntlDict : dictionary, optional
            loop control parameters in dictionary. The default is {}.
            Keys include {"step", "stepMin", "maxError", "loopLim", "relLim", "dfDelta", "printSum"}

        Returns
        -------
        None.

        """

        if not bool(self.loopCntl):
            self.loopCntl["step"] = 1             # pivot update step size control to ensure convergence
            self.loopCntl["stepMin"] = 1/256      # minimum step size for iterations to continue
            self.loopCntl["maxError"] = 1e12      # initialize max error variable computed at each iteration
            self.loopCntl["loopLim"] = 10000      # loop iteration max limit to stop iteration
            self.loopCntl["relLim"] = 0.01       # relative error target limit to stop iteration
            self.loopCntl["dfDelta"] = 0.000001   # derivative computation delta x
            self.loopCntl["printSum"] = False     # prints summary of PWM fit results
        # Update loop Control parameters
        for key in loopCntlDict:
            self.loopCntl[key] = loopCntlDict[key]

    def df(self, x):
        """
        Evaluates numerical derivative f'(x) at x values

        Parameters
        ----------
        x : double / np.array
            input x

        Returns
        -------
        ddx : double / np.array
            numerical derivative f'(x) evaluated at x.

        """
        dfDelta = self.loopCntl["dfDelta"]
        xClip = np.clip(x, self.xmin + dfDelta/2, self.xmax - dfDelta/2)
        ddx = (self.f(xClip + dfDelta/2) - self.f(xClip - dfDelta/2)) / dfDelta
        return ddx


    def fpwl(self, x):
        """
        returns y value evaluated at the corresponding input x for the given pwl function
        It is an approximation of the function  y = f(x)

        Parameters
        ----------
        x : double / np.array
           function input for which the function value is evaluated

        Returns
        -------
        y : TYPE
            function value for the given function input

        """
        # Find segment by comparing x to knots:
        k = np.clip(np.searchsorted(self.knot, x) - 1, 0, self.N-1)
        y = self.fknot[k] + self.slope[k] * (x - self.knot[k])
        return y


    def finvpwl(self, y) :
        """
        returns x value for a corresponding y for the given pwl functin
        It is an approximation of the function inverse x = f^-1(y)

        Parameters
        ----------
        y : double / np.array
            function value for which the function input is evaluated

        Returns
        -------
        x : double / np.array
            function input which evaluates to the function value y given the pwl function

        """
        # Find segment by comparing x to knots:
        slope = self.fknot[1:] - self.fknot[:-1]
        monotonicp = (slope.min() >= 0) # function appears to be monotonic positive
        monotonicn = (slope.max() <= 0) # function appears to be monotonic negative
        if monotonicp:  # fknot is in ascending order, find knot below y
            k = np.clip(np.searchsorted(self.fknot, y) - 1, 0, self.N-1)
        elif monotonicn:     # fknot is in descending order, find knot above y
            k = np.clip(np.searchsorted(-self.fknot, -y), 1, self.N) - 1
        else:
            print('Function not monotonic, inverse does not exist')
            k = np.clip(np.searchsorted(self.fknot, y) - 1, 0, self.N-1)
        x = self.knot[k] + ((y-self.fknot[k]) / self.slope[k])
        return x


    def fitPWL(self, knot0 = [], pivot0 = []) :
        """
        This is the method that fits the PWL to the function using a recursive gradient descent approach
        The pivot locations define the PWM as the tangents at the pivot
        The intersections of these tangents define the knots
        The errors at the knots are evaluated by comparing the tangent function to the targets functions
        3 types of max errors are supported: ey(vertical), ex(horizontal), en(normal)
        the sensitivity (gradient) of the errors are then computed relative to the pivots
        the pivots are then updated in the direction to reduce the error and with a step proportional to the gradient
        The recursion continues until all the errors are driven to the same relative magnitude (~ equal)
        This defines the PWL approximation to be used, and the knots are then modified to reduce the error magnitude in half
        This is the optimal minimax error that can be achieved with a N segment PWL.
        """

        # Copy variables to local name space
        f = self.f
        df = self.df
        xmin = self.xmin
        xmax = self.xmax
        N = self.N
        mode = self.mode

        # Copy/Initialize loop Control variables to local name space
        step = self.loopCntl["step"]           # pivot update step size control to ensure convergence
        stepMin = self.loopCntl["stepMin"]     # minimum step size for iterations to continue
        maxError = self.loopCntl["maxError"]   # initialize max error variable computed at each iteration
        loopLim = self.loopCntl["loopLim"]     # loop iteration max limit to stop iteration
        relLim = self.loopCntl["relLim"]       # relative error target limit to stop iteration
        dfDelta = self.loopCntl["dfDelta"]     # derivative computation delta x
        printSum = self.loopCntl["printSum"]   # prints summary of PWL fit results

        # Initialize N+1 Knots and N Pivots arrays
        if len(pivot0) == N:
            pivot = np.sort(np.clip(pivot0, xmin, xmax))
            knot = np.zeros(N+1)
            knot[1:N] = (pivot[0:N-1] + pivot[1:N]) / 2
            knot[0] = xmin
            knot[N] = xmax
        elif len(knot0) == N+1 :
            knot = np.sort(np.clip(knot0, xmin, xmax))
            pivot = (knot[0:N] + knot[1:N+1]) / 2
        else:
            knot = np.linspace(xmin, xmax, N+1) # N+1 knot locations in x axis as a np.array
            if N > 1:
                pivot = (knot[0:-1] + knot[1:])/2   # N pivot location in x axis as a np.array
            else:
                pivot = np.zeros(1)
                pivot[0] = (knot[0] + knot[1]) / 2

        # Function Evaluation
        gx = f(knot)                                      # Function values at knot
        slope = (gx[1:]-gx[:-1]) /  (knot[1:]-knot[:-1])  # function derivatives at knot
        self.monotonicp = (slope.min() >= 0) # function appears to be monotonic positive
        self.monotonicn = (slope.max() <= 0) # function appears to be monotonic negative
        if (N > 1):
            dslope = (slope[1:]-slope[:-1]) / (knot[2:]-knot[:-2]) # function
            self.concave = (dslope.min() >= 0) # function appears to be concave
            self.convex  = (dslope.max() <= 0) # function appears to be convex
        else:
            if (f(pivot[0]) <= (f(knot[0]) + f(knot[1]) / 2 )):
                self.concave = True # function appears to be concave
                self.convex = False
            else:
                self.convex  = True # function appears to be convex
                self.concave = False
        self.left = (self.monotonicp and self.convex) or (self.monotonicn and self.concave)  # PWL to the left of function, ex is negative
        self.right = (self.monotonicp and self.concave) or (self.monotonicn and self.convex) # PWL is to the right of dunction, ex is positive
        self.above = self.convex   # PWL is above function, ey is positive
        self.below = self.concave  # PWL is below function, ey is negative

        # Initialize arrays
        ex = np.ones(N+1) * maxError  # iterated PWl approximation error at knot
        ey = np.ones(N+1)  * maxError # iterated PWl approximation error at knot
        en = np.zeros(N+1)  # iterated PWl approximation error at knot
        delta = np.zeros(N) # iterated pivot update change
        lastPivot = np.zeros(N) # save old pivot in case iteration needs to revert

        # Recursive Loop, each cycle updates pivot[] such that
        loopCnt = 0         # counts number of recursive iterations algorithm
        stopIter = False
        lastMaxError =  maxError + 1

        while (stopIter is False):
            if not (self.concave or self.convex):
                print("function is not concave or convex ")
                stopIter = True
                break

            # Evaluate f(x) and f'(x) at pivots
            ft = f(pivot)
            dft = df(pivot)

            # Find knot locations given pivots as intersection of tangent lines at pivots
            for i in range(1, N):
                knot[i] = ( (ft[i-1] - ft[i] + dft[i]*pivot[i] - dft[i-1]*pivot[i-1]) /
                            (dft[i] - dft[i-1]) )

            # Find knot coordinates (knot, gx)
            for i in range(N):
                gx[i] = dft[i]*(knot[i]-pivot[i]) + ft[i]
            gx[N] = dft[N-1]*(knot[N]-pivot[N-1]) + ft[N-1]

            # Find error at knots
            # Vertical error ey
            ey = gx -  f(knot)

            # Horizontal error ex using triangular approximation with ey and slope m[i]
            m = self.df(knot)

            # If derivative is 0/undefined, use steeper slope from adjacent segments
            # For first/last knots, use first/last segment slope

            ex[0] = - ey[0] / (dft[0])
            ex[N] = - ey[N] / (dft[N-1])

            for i in range(1, N):
                if (m[i] == 0):
                    if abs(dft[i-1]) > abs(dft[i]) :
                         ex[i] = - ey[i] / (dft[i-1])
                    else:
                         ex[i] = - ey[i] / (dft[i])
                else:
                    ex[i] = - ey[i] / (m[i])

            # Normal error en using triangular approximation with ey and slope m[i]
            en[0] = np.abs(ey[0]) * (1 / (1 + dft[0]**2)**0.5)
            en[N] = np.abs(ey[N]) * (1 / (1 + dft[N-1]**2)**0.5)

            for i in range(1,N):
                #en[i] = np.abs(ex[i] * ey[i]) / (ex[i]**2 + ey[i]**2)**0.5
                en[i] = np.abs(ey[i]) * (1 / (1 + m[i]**2)**0.5)

            # Choose error type depending on mode
            if (mode == "ey") :
                err = np.copy(ey)
            elif (mode == "ex") :
                err = np.copy(ex)
            elif (mode == "en") :
                err = np.copy(en)
            else :
                print("Mode not properly set")

            # Find maximum, minimum, and relative error
            maxError = np.absolute(err).max()
            minError = np.absolute(err).min()
            relError = (maxError - minError) / maxError # relative error is 0 when minError = maxError

            # if error is increasing/diverging then reduce step size and revert to previous step
            if (maxError > lastMaxError) :
                if step <= stepMin:
                    print ("Error is diverging at stepsize: ", step)
                    stopIter = True
                step = step / 2
                pivot = np.copy(lastPivot) # revert pivots
                print ("reverting at step", loopCnt, maxError)

            # if error is decreasing/converging, update pivots and continue descent until error Limit or loop count limit is met
            else:
                if (relError < relLim):
                    print ("Relative Error ", relError, "met at loopCnt ", loopCnt)
                    stopIter = True
                elif (loopCnt > loopLim):
                    print ("LoopCnt limit ", loopCnt, "reached at relError ", relError)
                    stopIter = True
                else:
                    # Update Pivots
                    for i in range(N):
                        delta[i] = (err[i+1] - err[i]) / ( (err[i+1]/(knot[i+1]-pivot[i])) + (err[i]/(pivot[i] - knot[i])) )
                    lastMaxError = maxError
                    lastPivot = np.copy(pivot)
                    pivot += step * delta
                    loopCnt += 1

        # Store final iterated pivot and knot locations, use to define the PWL function
        self.err = np.copy(err)
        self.pivot = np.copy(pivot)
        self.knot = np.copy(knot)  # unadjusted knot location
        self.fknot = np.copy(gx)

        # Adjust knot locations to minimize error depending on error mode
        if (mode == "ey"):
            self.fknot -= ey/2     # move knot vertically to reduce error in half to make it symmetric
        elif (mode == "ex"):
            self.knot -= ex/2      # move knot horizontally to reduce error in half to make it symmetric
        elif (mode == "en"):
            errorx = ex * ey**2 / (ex**2 + ey**2)
            errory = ey * ex**2 / (ex**2 + ey**2)
            self.knot -= errorx / 2    # move knots "inward" to reduce normal error in half and make it symmetric
            self.fknot -= errory / 2

        self.error = maxError /2
        # Calculate PWL segment slopes based on adjusted knot locations
        self.slope = (self.fknot[1:] - self.fknot[:-1]) / ((self.knot[1:] - self.knot[:-1]))
        # Fit first and last knot to xmin, xmax
        self.fknot[0] = self.fknot[1] - self.slope[0] * (self.knot[1] - self.xmin)
        self.knot[0] = self.xmin
        self.fknot[N] = self.fknot[N-1] + self.slope[N-1] * (self.xmax - self.knot[N-1])
        self.knot[N] = self.xmax

        # Summary of algorithm results
        if printSum :
            print("function: ", [self.monotonicp, self.monotonicn, self.concave, self.convex])
            print("relError, maxError: ", relError, maxError)
            print("step, loopCnt: ",step, loopCnt)
            print()
            print("pivots \n", pivot)
            print("f(pivots) \n", ft)
            print("f'(pivots) \n", dft)
            print()
            print("knots \n", self.knot)
            print("g(knots) \n", self.fknot)
            print()
            print("ex(knots) \n", ex)
            print("ey(knots) \n", ey)
            print("en(knots) \n", en)
            print("delta \n", delta)


# Knot Initialization Functions to aid convergence

def knot_rescale(knot, xmin, xmax):
    # Scales knots into [xmin,xmax] range
    scale = (xmax - xmin) / (knot[-1] - knot[0])
    sknot = scale * knot
    offset = xmin - sknot[0]
    return (sknot + offset)

def knot_double(knot, xmin, xmax, k=0):
    # Combine N+1 knots to 2N/2N+1/2N+2 knots
    knot0 = np.zeros(2 * knot.size - 1)
    knot0[0::2] = knot
    knot0[1::2] = (knot[0:-1] + knot[1:]) / 2

    if (k < -1 or k > 1):
        print("knot0 size is not compatible")
    else:
        # Add a knot
        if (k == 1):
            knot0 = np.append(knot0, (2 * knot0[-1] - knot0[-2]) )
        # Remove a knot
        if (k == -1):
            knot0 = knot0[0:-1]

        # Rescale
        knot0 = knot_rescale(knot0, xmin, xmax)
    return knot0

def knot_errscale(knot, err, xmin, xmax):
    # Scales knots into [xmin,xmax] range with spacing inversely proportional to errors
    errmean = err.mean()
    scale = (err[1:] + err[:-1] + 0.01 * errmean) / (2.01 * errmean) # in case 0 error don't perfectly scale
    dx = (knot[1:] - knot[:-1]) / scale
    knot0 = np.append(knot[0], knot[0] + np.cumsum(dx))
    knot0 = knot_rescale(knot0, xmin, xmax)
    return knot0

# Display Functions
def pwlfDisplay(f1: pwl, pts = 40001, fn ="", title=""):
    # Display the PWL function
    x = np.linspace(f1.xmin, f1.xmax, pts)
    plt.plot(x, f1.fpwl(x),'c-', label="pwl")
    plt.plot(f1.knot, f1.fknot, 'bo', label="knots")
    plt.plot(f1.pivot, f1.f(f1.pivot), 'r+', label = "pivots")
    if title == "":
        title = "pwlfDisplay() - PWL Approximation of f(x)"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("PWL(x)")
    plt.legend()
    if fn != "":
        plt.savefig(fn)
        print("Figure saved to ", fn)
    plt.show()

def pwlfErrorDisplay(f1: pwl, pts = 40001, fn ="", title=""):
    # Compare the PWL function error relative to the target function
    x = np.linspace(f1.xmin, f1.xmax, pts)
    plt.plot(x, (f1.fpwl(x) - f1.f(x)))
    if title == "":
        title = "pwlfErrorDisplay() - PWL Approximation Error for f(x)"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("PWL(x) - f(x)")
    if fn != "":
        plt.savefig(fn)
        print("Figure saved to ", fn)
    plt.show()

def pwlfinvDisplay(f1: pwl, pts=40001, fn ="", title=""):
    # Display the inverse PWL function
    y = np.linspace(min(f1.f(f1.xmin),f1.f(f1.xmax)), max(f1.f(f1.xmin),f1.f(f1.xmax)), pts)
    plt.plot(y, f1.finvpwl(y), 'g-', label="inv pwl")
    plt.plot(f1.fknot, f1.knot, 'bo', label="knots")
    plt.plot(f1.f(f1.pivot), f1.pivot, 'r+', label = "pivots")
    if title == "":
        title = "pwlfinvDisplay() PWL Approximation of f$^{-1}$(y)"
    plt.title(title)
    plt.xlabel("y")
    plt.ylabel("inv PWL(y)")
    plt.legend()
    if fn != "":
        plt.savefig(fn)
        print("Figure saved to ", fn)
    plt.show()

def pwlfinvErrorDisplay(f1: pwl, finv, pts=40001, fn ="", title=""):
    # Compare the PWL function error relative to the target function
    if f1.monotonicp or f1.monotonicn :
        y = np.linspace(min(f1.fknot), max(f1.fknot), pts)
        plt.plot(y, (f1.finvpwl(y) - finv(y)))
        if title == "":
            title = "pwlfinvErrorDisplay() - inv PWL Approximation Error for f$^{-1}$(y)"
        plt.title(title)
        plt.xlabel("y")
        plt.ylabel("inv PWL(y) - f$^{-1}$(y)")
        if fn != "":
            plt.savefig(fn)
            print("Figure saved to ", fn)
        plt.show()
    else:
        print("Function not monotonic, inverse does not exist")

def pwlfTangentDisplay(f1: pwl, pts = 40001, fn ="", title=""):
    # Plot the PWL function tangents at pivots
    x = np.linspace(f1.xmin, f1.xmax, pts)
    for i in range(f1.N - 1):
        y = f1.f(f1.pivot[i]) + f1.df(f1.pivot[i]) * (x - f1.pivot[i])
        plt.plot(x, y)
    plt.plot(x, f1.f(x), linestyle='--', color='black')
    if title == "":
        title = "pwlfTangentDisplay() - PWL approximation of f(x), Tangents Visualization"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("PWL(x)")
    plt.xlim(f1.xmin, f1.xmax)
    plt.ylim(min(f1.fknot), max(f1.fknot))
    if fn != "":
        plt.savefig(fn)
        print("Figure saved to ", fn)
    plt.show()

def pwlKnotDetailDisplay(f1: pwl, index=1, pts = 2001, fn ="", title=""):
    # Display the PWL function Detail around a knot
    i = np.clip(index, 1, f1.N-1)
    xmin = f1.knot[i-1]
    xmax = f1.knot[i+1]
    ymin = min(f1.fknot[i-1], f1.fknot[i+1])
    ymax = max(f1.fknot[i-1], f1.fknot[i+1])
    x = np.linspace(xmin, xmax, pts)
    tang1 = f1.f(f1.pivot[i-1]) + f1.df(f1.pivot[i-1]) * (x - f1.pivot[i-1])
    tang2 = f1.f(f1.pivot[i]) + f1.df(f1.pivot[i]) * (x - f1.pivot[i])
    plt.plot(x, f1.f(x), 'g-', label = "f(x)")
    plt.plot(x, f1.fpwl(x),'c-', label="pwl")
    plt.plot(x, tang1, '--')
    plt.plot(x, tang2, '--')
    plt.plot(f1.knot[i-1:i+2], f1.fknot[i-1:i+2], 'bo', label=("knots " + str(i-1) + "-" + str(i+1)))
    plt.plot(f1.pivot[i-1:i+1], f1.f(f1.pivot[i-1:i+1]), 'r+', label = "pivots")

    if title == "":
        title = "pwlKnotDetailDisplay() - PWL Approximation of f(x), Knot Detail"
    plt.title(title)
    plt.xlim(xmin, xmax)
    # plt.ylim(ymin, ymax)
    plt.xlabel("x")
    plt.ylabel("PWL(x)")
    plt.legend()
    if fn != "":
        plt.savefig(fn)
        print("Figure saved to ", fn)
    plt.show()

def pwlPivotDisplay(f1 : pwl, fn ="", title=""):
    # Plot Knots and Pivots vs indices
    index = np.linspace(0, f1.N, f1.N+1)
    plt.plot(f1.knot, index, "b-o", label="Knots")
    plt.plot(f1.pivot, index[:-1], "r-+", label="Pivots")
    if title == "":
        title = "pwlKnotLocationDisplay() - PWL approximation of f(x), Pivot and Knot Locations vs Index"
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("index")
    plt.ylim(0, f1.N)
    plt.legend()
    if fn != "":
        plt.savefig(fn)
        print("Figure saved to ", fn)
    plt.show()

def pwlPivotAngleDisplay(f1: pwl, fn ="", title=""):
    # plot knot locations as angle in polar coordinates
    # Interesting case when f(x) = sqrt(1 - x^2), because pivots should be linearly spaced
    index = range(f1.N+1)
    plt.plot(np.arctan(f1.f(f1.pivot)/f1.pivot)*(180/np.pi) % 180, index[:-1], "r-+", label="Pivot Angles")
    plt.plot(np.arctan(f1.f(f1.knot)/f1.knot)*(180/np.pi) % 180, index, "b-o", label="Knot Angles")
    if title == "":
        title = "pwlPivotAngleDisplay() - PWL approximation of f(x), Knot Angles"
    plt.title(title)
    plt.ylabel("index")
    plt.xlabel("knot angle")
    plt.legend()
    if fn != "":
        plt.savefig(fn)
        print("Figure saved to ", fn)
    plt.show()
