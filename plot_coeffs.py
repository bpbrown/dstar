"""
Plot coeffs from joint analysis files.

Usage:
    plot_coeffs.py <files>... [options]

Options:
    --fields=<fields>    Fields to extract coeffs of [default: ρu]
    --output=<output>    Output directory; if blank a guess based on likely case name will be made
    --dpi=<dpi>          dpi for image files (if png) [default: 300]
"""
import logging
logger = logging.getLogger(__name__.split('.')[-1])

for system in ['matplotlib', 'h5py']:
    dlog = logging.getLogger(system)
    dlog.setLevel(logging.WARNING)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import h5py

if __name__ == "__main__":
    import pathlib
    from docopt import docopt
    from dedalus.tools import post

    import logging
    logger = logging.getLogger(__name__.split('.')[-1])

    args = docopt(__doc__)
    if args['--output'] is not None:
        output_path = pathlib.Path(args['--output']).absolute()
    else:
        data_dir = args['<files>'][0].split('/')[0]
        output_path = pathlib.Path(data_dir).absolute()
    # Create output directory if needed
    if not output_path.exists():
        output_path.mkdir()
    logger.info("output to {}".format(output_path))

    dpi = float(args['--dpi'])

    fields = args['--fields'].split(',')

    def accumulate_files(filename,start,count,file_list):
        if start==0:
            file_list.append(filename)

    file_list = []
    post.visit_writes(args['<files>'],  accumulate_files, file_list=file_list)
    logger.info(file_list)

    reg = 2
    times = np.array([])
    data = []
    for file in file_list:
        print('reading in {:s}'.format(file))
        f = h5py.File(file, 'r')
        t = np.array(f['scales/sim_time'])
        times = np.concatenate((times, t))
        for k in range(len(t)):
            for i, field in enumerate(fields):
                task = f['tasks'][field]
                m = task.dims[2][0][:]
                ell = task.dims[3][0][:]
                n = task.dims[4][0][:]
                mask = ((m == 0) + (m == 1))*(ell == 1)*(n == 0)
                data.append(task[k][reg][mask])
        f.close()
    data = np.array(data).T

    fig, ax = plt.subplots(nrows=2, sharex=True)
    print('m:{}, ell:{}, n:{}'.format(m[mask], ell[mask], n[mask]))
    ax[0].plot(times, data[2,:], label=r'$\ell=1,m=1s,n=0$')
    ax[0].plot(times, data[3,:], label=r'$\ell=1,m=1c,n=0$')
    ax[0].plot(times, data[1,:], label=r'$\ell=1,m=0c,n=0$')
    ax[1].plot(times, np.abs(data[2,:]), label=r'$\ell=1,m=1s,n=0$')
    ax[1].plot(times, np.abs(data[3,:]), label=r'$\ell=1,m=1c,n=0$')
    ax[1].plot(times, np.abs(data[1,:]), label=r'$\ell=1,m=0c,n=0$')
    ax[1].set_xlabel('time')
    ax[0].set_ylabel(r'$\rho u^{0}$')
    ax[1].set_ylabel(r'$|\rho u^{0}|$')
    ax[1].set_yscale('log')
    ax[0].legend()
    ax[1].legend()
    fig.savefig(data_dir+'/angular_momentum_coeffs.png', dpi=300)
