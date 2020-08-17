# This import registers the 3D projection, but is otherwise unused.
from datetime import datetime
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from mpmath import * # zeta, zetaroot, etc

"""Generic grid searches
"""

verbose = False
store_results = True
store_plots = True
show_plots = False

def zeta_grid_search(g, # function to evaluate whether grid may contain root
                    loops = 100, # number of imaginary steps, including infinity
                    r = [0,0.25,0.375,0.4375,0.5], # real break points
                    h = 1, # imaginary step size
                    t0 = 5700, # Starting imaginary
                    g_args = {}): # Arguements for g
    """[summary]
    Systematically search grid squares in a search grid.
    The search grid partitions the search strip.
    The grid is defined by t0, r, h and loops.
    The grid is: -
    Reals: 0 = r0 < ... < rn = 0.5.
    Imags: t0 < t0+h < ...t0+loops*h.
    Each grid square is passed to the function g.
    g = g(grid) where grid=[s00,s01,s10,s11]
    where sij are complex numbers and
    s00: bottom left grid point
    s01: bottom right grid point
    s10: top left grid point
    s11: top right grid point
    g must return a dictionary with two members.
    'keep': True/False.
    'result': A result.
    If keep is true the grid may contain a zero and
    is appended to the list containing the result of this function.
    """

    s0 = [complex(f,t0) for f in r]
    print(">> Start of search: " + str(datetime.now()))
    print(">> real", r)
    keep_grids = []
    if store_results: 
        z0s = open('results/z0_' + str(datetime.date(datetime.now()))+ '.txt', 'a+')
    for n in range(loops):
        t1 = t0 + h
        if verbose:
            print(">> loop", n+1, " of ", loops, ", imag [", t0, ", ", t1, "]")
        s1 = []
        for f in r:
            s = complex(f,t1)
            s1 += [s]
        grids = [[s0[i], s0[i + 1], s1[i], s1[i + 1]] 
                for i in range(len(r) - 1)]
        for grid in grids:
            #print("Grid: ", grid)
            ret = g(grid, **g_args)
            #print("Result: ", ret)
            if ret['keep']:
                keep_grids.append({'grid': grid, 'result': ret['val']})
                noncrit = ' NON-CRIT!' if grid[3].real != 0.5 else ''
                if store_results:
                    rec =  str(grid) + '|' + str(ret['val']) + '|' + noncrit + '\n'
                    #print(rec)
                    z0s.write(rec)
                print(">>> loop", n+1, " of ", loops,
                    "\n Grid: ", grid, noncrit,
                    "\n Result: ", ret['val'],
                    "\n Time: " + str(datetime.now()))
        t0 = t1
        s0 = s1
    print(">> End of search: " + str(datetime.now()))
    if verbose: print(">> Grids: \n" + str(keep_grids))
    if store_results: z0s.close()
    return keep_grids

def zeta_vals(grid, divs = 20):
    """ 
    Get the values of zeta over a grid.
    """
    # Make the grid
    x = np.linspace(grid[0].real, grid[-1].real, divs)
    y = np.linspace(grid[0].imag, grid[-1].imag, divs)
    X, Y = np.meshgrid(x, y)
    zs = np.array([zeta(complex(x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    # Calc zeta
    return {'X':X, 'Y':Y, 'Z':Z}

def corner_crossings(grid):
    ret = {}
    ret['keep'] = False
    ret['val'] = None
    # Get zeta at the corners
    z_val = [zeta(s) for s in grid]
    for p in range(4):
        for q in range(p+1,4):
            re = z_val[p].real * z_val[q].real
            im = z_val[p].imag * z_val[q].imag
            if re < 0 and im < 0:
                # See if real or imag cross zero.
                # If they do, do a zero search.
                ret['keep'] = True
                ret['val'] = z_val
                return ret
    return ret

def modulus_tiny(grid, divs=10, abs_min=1e-1):
    ret = {}
    ret['keep'] = False
    ret['val'] = None
    # Get zeta at subgrid points
    zs = zeta_vals(grid, divs = divs)
    # Check if tiny modulus occurs 
    zs_abs = [float(abs(z)) for z in np.ravel(zs['Z'])]
    zs_min = min(zs_abs)
    if zs_min < abs_min:
        # Get the point of the minimum
        min_at = None
        #print("zs_abs", zs_abs)
        #print("zs[Z]", zs['Z'])
        for i in range(len(zs['Z'])):
            for j in range(len(zs['Z'][i])):
                row_val = float(abs(zs['Z'][i][j]))
                if zs_min == row_val:
                    min_at = complex(zs['X'][i][j], zs['Y'][i][j])
                    break
            if min_at: break
        # Set result
        ret['keep'] = True
        ret['val'] = [zs_min, min_at]
        return ret
    return ret

def tiny_crossing(grid, divs=20, abs_min=1e-3):
    ret = corner_crossings(grid)
    if ret['keep']:
        ret1 = ret['val']
        ret = modulus_tiny(grid, divs=divs, abs_min=abs_min)
        if ret['keep']:
            ret2 = ret['val']
            ret['val'] = ret2
            return ret
    return ret

def some_near_zeros():
    n0s = []
    n0s.append([(0.417+5884j), (0.5+5884j), (0.417+5885j), (0.5+5885j)])
    n0s.append([(0.417+5998j), (0.5+5998j), (0.417+5999j), (0.5+5999j)])
    print(n0s)

"""Plotting functions
"""

def zeta_plot_line(re = 0.5, im0 = 0, im1 = 30, steps = 1000):
    x = np.linspace(im0, im1, steps)  # Create a list of evenly-spaced numbers over the range
    z = [zeta(complex(re, t)) for t in x] # Calc zeta on line
    fignm = 'Zeta Along Re(z)=' + str(re) 
    plt.figure(num=fignm)
    plt.plot(x, [a.real for a in z], label='real') # Plot the real part 
    plt.plot(x, [a.imag for a in z], label='imag') # Plot the imaginary part
    plt.plot(x, [float(abs(a)) for a in z], label='mod') # Plot the modulus
    plt.axhline(0, color='grey')
    #plt.axvline(0, color='grey')
    plt.legend()
    plt.title(fignm)
    plt.show() # Display the plot

def zeta_plot_grid(grid, divs = 20):
    """ 
    Get the values of zeta over a grid.
    """
    # Make the grid
    grid_vals = zeta_vals(grid, divs=divs)
    X = grid_vals['X']
    Y = grid_vals['Y']
    Z = grid_vals['Z']
    zs = np.ravel(Z)

    # Make the origin plane
    orig = np.array([0 for x,y in zip(np.ravel(X), np.ravel(Y))])
    O = orig.reshape(Z.shape)

    # Calc zeta
    zs_real = np.array([float(z.real) for z in zs])
    zs_imag = np.array([float(z.imag) for z in zs])
    zs_abs = np.array([float(abs(z)) for z in zs])

    # Initialise plots
    fignm = 'Zeta on Grid ' + str(grid) 
    fig = plt.figure(num=fignm, figsize=plt.figaspect(1/3))

    # Plot the real part
    Z_real = zs_real.reshape(Z.shape)
    ax = fig.add_subplot(131, projection='3d', title='Re(z)')
    ax.set_xlabel('real')
    ax.set_ylabel('imag')
    ax.plot_surface(X, Y, Z_real)
    ax.plot_surface(X, Y, O, color='grey', alpha=0.6)
    ax.set_zlabel('real part')

    # Plot the imaginary part
    Z_imag = zs_imag.reshape(Z.shape)
    ax = fig.add_subplot(132, projection='3d', title='Im(z)')
    ax.set_xlabel('real')
    ax.set_ylabel('imag')
    ax.plot_surface(X, Y, Z_imag)
    ax.plot_surface(X, Y, O, color='grey', alpha=0.6)
    ax.set_zlabel('imaginary part')

    # Plot the modulus
    Z_abs = zs_abs.reshape(Z.shape)
    ax = fig.add_subplot(133, projection='3d', title='Mod(z)')
    ax.set_xlabel('real')
    ax.set_ylabel('imag')
    ax.plot_surface(X, Y, Z_abs)
    ax.plot_surface(X, Y, O, color='grey', alpha=0.6)
    ax.set_zlabel('modulus')

    if store_plots: plt.savefig('results/zeta' + str(grid) + '.png')
    if show_plots: plt.show()
    
def plot_root_grids():
    root_grids = zeta_root_grid_search()
    for root_grid in root_grids:
        #print(root_grid[0][0],root_grid[-1][0])
        zeta_plot_grid(root_grid[0][0],root_grid[-1][0])

    """Play stuff
    """

def np_play():
    """ 
    Get the values of zeta over a grid.
    """
    # Make the grid
    grid = [(0.375+5706j), (0.417+5706j), (0.375+5707j), (0.417+5707j)] 
    divs = 4
    x = np.linspace(grid[0].real, grid[-1].real, divs)
    y = np.linspace(grid[0].imag, grid[-1].imag, divs)
    print("x,y", x, y)
    X, Y = np.meshgrid(x, y)
    print("X Y meshgrid", X, Y)

    # Calc zeta
    print("X Y ravel", np.ravel(X), np.ravel(Y))
    print("X Y zip", zip(np.ravel(X), np.ravel(Y)))
    zs = np.array([zeta(complex(x,y)) for x,y in zip(np.ravel(X), np.ravel(Y))])
    print("zs", zs)
    Z = zs.reshape(X.shape)
    print("Z", Z)

    """Pointy end
    """

def main():
    #print(zeta_vals([5707j, (0.25+5707j), 5708j, (0.25+5708j)],3))
    #zeta_grid_search(corner_crossings)
    #zeta_grid_search(modulus_tiny, g_args={'divs':10, 'abs_min':1e-1})
    zeta_grid_search(tiny_crossing, 
                    loops=10, 
                    t0=5707,
                    g_args={'divs':10, 'abs_min':1e-3})
    #zeta_plot_line()
    #zeta_plot_grid([(0.417+5884j), (0.5+5884j), (0.417+5885j), (0.5+5885j)],20)
    #np_play()
    #print(float(abs(zeta(0.5+5706.111111111111j))))
    #some_near_zeros()
    pass

if __name__ == '__main__':
    main()