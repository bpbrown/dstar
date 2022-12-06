"""
Plot slices from joint analysis files.

Usage:
    plot_coeffs.py <files>... [options]

Options:
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
    from dedalus.tools.parallel import Sync

    import logging
    logger = logging.getLogger(__name__.split('.')[-1])

    args = docopt(__doc__)
    if args['--output'] is not None:
        output_path = pathlib.Path(args['--output']).absolute()
    else:
        data_dir = args['<files>'][0].split('/')[0]
        data_dir += '/frames/'
        output_path = pathlib.Path(data_dir).absolute()
    # Create output directory if needed
    with Sync() as sync:
        if sync.comm.rank == 0:
            if not output_path.exists():
                output_path.mkdir()
    logger.info("output to {}".format(output_path))

    dpi = float(args['--dpi'])

    fields = ['ρu'] #fields = args['--fields'].split(',')

    def accumulate_files(filename,start,count,file_list):
        if start==0:
            file_list.append(filename)

    file_list = []
    post.visit_writes(args['<files>'],  accumulate_files, file_list=file_list)
    logger.info(file_list)
    if len(file_list) > 0:
        for file in file_list:
            print('reading in {:s}'.format(file))
            f = h5py.File(file, 'r')
            t = np.array(f['scales/sim_time'])
            print(f['scales/write_number'][:])
            for k in range(len(t)):
                for i, field in enumerate(fields):
                    time = t
                    center_zero=False
                    title = field
                    task = f['tasks'][field]
                    # phi = task.dims[1][0][:]
                    # theta = task.dims[2][0][:]
                    # r = task.dims[3][0][:]
                    data_slices = (k, slice(None), slice(None), slice(None), slice(None))
                    data = task[data_slices]
                    i=2
                    print('reg ', i, data[i,0:4,1,0])
