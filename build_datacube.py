"""
Build datacube of azimuthally averaged quantities.

Usage:
     build_datacube.py <files>... [options]
     build_datacube.py --case=<case> [options]

Options:
     <files>...         List of files to compute azavg over.
     --case=<case>      case to build data cube for; alternative use pattern to specifying files.

     --output=<output>  Location to save datacube; if blank, a sensible guess based on <files> will be made.

     --datacube=<datacube>  Pre-existing time-latitude datacube

"""
import numpy as np
from mpi4py import MPI
from dedalus.tools import post
import logging
logger = logging.getLogger(__name__)

def read_data(files):
    start_time = time.time()

    def accumulate_files(filename,start,count,file_list):
        if start==0:
            file_list.append(filename)

    file_list = []
    post.visit_writes(files,  accumulate_files, file_list=file_list)
    logger.debug(file_list)

    data = {}
    times = None
    r = None
    theta = None

    color = int(len(file_list) > 0)
    comm = MPI.COMM_WORLD.Split(color=color)
    rank = comm.rank
    size = comm.size
    if color:
        for file in file_list:
            logger.debug("opening file: {}".format(file))
            f = h5py.File(file, 'r')
            data_slices = (slice(None), 0, slice(None), slice(None))
            for task in f['tasks']:
                if task[0] == '<' and task[-1] == '>': #azavgs denoted with '<q>'
                    logger.info("task: {}".format(task))
                    if task in data:
                        data[task] = np.append(data[task], f['tasks'][task][data_slices], axis=0)
                    else:
                        data[task] = np.array(f['tasks'][task][data_slices])
                    if r is None or theta is None:
                        theta = f['tasks'][task].dims[2][0][:]
                        r = f['tasks'][task].dims[3][0][:]
            if times is None:
                times = f['scales/sim_time'][:]
            else:
                times = np.append(times, f['scales/sim_time'][:])
            f.close()
    else:
        data = {'zero':np.zeros((1,1,1))}
        times = np.zeros(1)

    comm.Barrier()

    n_global_time = np.array(0)
    n_global_data = np.array(0)

    data_set = data
    global_data_set = {}

    comm.Reduce([np.array(times.size), MPI.INT], [n_global_time, MPI.INT], op=MPI.SUM, root=0)
    if rank == 0:
        n_times_each = np.empty([size], dtype=np.int)
        global_time = np.empty([n_global_time], dtype=np.float64)
    else:
        n_times_each = None
        global_time = None
    comm.Gather(np.array(times.size), n_times_each, root=0)
    if rank == 0:
        send_counts = tuple(n_times_each)
        displacements = tuple(np.append(np.zeros(1, dtype=np.int), np.cumsum(n_times_each))[0:-1])
    else:
        send_counts = None
        displacements = None
    comm.Gatherv(times.astype(np.float64), [global_time, send_counts, displacements, MPI.DOUBLE], root=0)

    for task in data_set:
        data = data_set[task]
        comm.Reduce([np.array(data.size), MPI.INT], [n_global_data, MPI.INT], op=MPI.SUM, root=0)
        if rank == 0:
            logger.info("{}: n_time = {}, n_data = {}".format(task, n_global_time, n_global_data))
            n_data_each = np.empty([size], dtype=np.int)
            global_data = np.empty([n_global_time,]+list(data[0,:,:].shape),dtype=np.float64)
            logger.debug("{}, {}, {}".format(data.shape, global_time.shape, global_data.shape))
        else:
            n_data_each = None
            global_data = None

        comm.Gather(np.array(data.size), n_data_each, root=0)

        if rank == 0:
            displacements = tuple(np.append(np.zeros(1, dtype=np.int), np.cumsum(n_data_each))[0:-1])
            send_counts = tuple(n_data_each)
        comm.Gatherv(data.astype(np.float64), [global_data, send_counts, displacements, MPI.DOUBLE], root=0)
        global_data_set[task] = global_data
    end_time = time.time()
    logger.info("time to build dataset {:g}sec".format(end_time-start_time))
    return global_data_set, global_time, theta, r

if __name__ == "__main__":
    import h5py
    import pathlib
    import time
    import glob
    import dedalus.public as de # parse dedalus.cfg file
    import logging
    logger = logging.getLogger(__name__.split('.')[-1])


    from docopt import docopt
    args = docopt(__doc__)
    if args['--output'] is not None:
        output_path = pathlib.Path(args['--output']).absolute()
    else:
        if len(args['<files>']) > 0:
            data_dir = args['<files>'][0].split('/')[0]
        elif args['--datacube']:
            data_dir = args['--datacube'].split('/')[0]
        elif args['--case']:
            data_dir = args['--case']
        else:
            data_dir = '.'
        data_dir += '/'
        output_path = pathlib.Path(data_dir).absolute()

    if args['--datacube']:
        datacube_filename = args['--datacube']
    else:
        datacube_filename = data_dir+'time_lat_datacube.h5'

    start_time = time.time()
    if args['--case']:
        file_glob = args['--case']+'/slices/slices_s*.h5'
        files = glob.glob(file_glob)
    else:
        files = args['<files>']
    from dedalus.tools.general import natural_sort
    files = natural_sort(files)

    data, times, theta, r = read_data(files)

    if MPI.COMM_WORLD.rank == 0:
        f_cube = h5py.File(datacube_filename,'w')
        f_cube['scales/r'] = r
        f_cube['scales/theta'] = theta
        f_cube['scales/sim_time'] = times
        for task in data:
            f_cube['tasks/{:s}'.format(task)] = data[task]
        f_cube.close()
        end_time = time.time()
        logger.info("time to build datacube {:g}sec".format(end_time-start_time))
