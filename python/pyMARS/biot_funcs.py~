import numpy as np
import time, multiprocessing, os
import pickle, itertools
import matplotlib.pyplot as pt
import mpl_toolkits.mplot3d.axes3d as p3

def surfmn_coil_points(ax):
    coil_details = np.loadtxt(file('/home/srh112/code/pyMARS/other_scripts/I-coil_geom.txt','r'))
    coil_point_list = []
    for i in range(0, coil_details.shape[0],14):
        current = np.zeros((15,3),dtype=float)
        current[0:14,:] = coil_details[i:i+14,:]
        current[14,:] = coil_details[i,:]
        ax.plot(current[:,0],current[:,1],current[:,2],'o-')
        coil_point_list.append(current)
        
    return coil_point_list


class biot_sav_worker(object):#multiprocessing.Process):
    def __init__(self, task_queue, result_queue):
        print os.getpid(), ' initialised worker'
        #multiprocessing.Process.__init__(self)
        self.task_queue = task_queue
        self.result_queue = result_queue

        #def run(self):
        #proc_name = self.name
        while True:
            next_task = self.task_queue.get()
            if next_task is None:
                # Poison pill means shutdown
                print os.getpid(), ' Exiting'
                self.task_queue.task_done()
                break
            #print '%s: %s' % (proc_name, next_task)
            print os.getpid(), ' starting a calc'
            answer = next_task.runn()#.run_calc()
            self.task_queue.task_done()
            self.result_queue.put(answer)
            print os.getpid(), ' finished, result put in queue'
        return


class biot_sav_job(object):
    def __init__(self, coil_points, xyz_X, xyz_Y, xyz_Z, coil_currents, dist_thresh):
        self.coil_points = coil_points
        self.xyz_X = xyz_X
        self.xyz_Y = xyz_Y
        self.xyz_Z = xyz_Z
        self.coil_currents = coil_currents
        self.dist_thresh = dist_thresh

    def runn(self):
        #def run_calc(self):
        for tmp in range(1,len(self.coil_points)):
            dl = self.coil_points[tmp] - self.coil_points[tmp-1]
            dl_loc = (self.coil_points[tmp] + self.coil_points[tmp-1])/2.
            points = np.transpose(np.array([self.xyz_X.flatten() - dl_loc[0], self.xyz_Y.flatten() - dl_loc[1], self.xyz_Z.flatten() - dl_loc[2]]))
            currents = np.transpose(np.array([self.xyz_X.flatten()*0+dl[0], self.xyz_Y.flatten()*0+dl[1], self.xyz_Z.flatten()*0+dl[2]]))*self.coil_currents
            mag_r = np.sqrt(points[:,0]**2+points[:,1]**2+points[:,2]**2)
            mag_r = np.transpose(np.array([mag_r,mag_r,mag_r]))
            r_hat = points/mag_r
            low_values_indices = mag_r < self.dist_thresh  # Where too close to I-coil
            mag_r[low_values_indices] = self.dist_thresh  # Replace with min dist_thresh
            if tmp==1:
                B = np.cross(currents, r_hat)*10**(-3)/(mag_r**2) #10^-3 is from G/kA and mu_0
            else:
                B += np.cross(currents, r_hat)*10**(-3)/(mag_r**2) #10^-3 is from G/kA and mu_0
        return B


def basic_calculation_wrapper(arguments):
    print 'started wrapper'
    return basic_calculation(*arguments)

def basic_calculation(coil_points, xyz_X, xyz_Y, xyz_Z, coil_currents, dist_thresh):
    '''
    This runs the basic biot-savart calculation for a single I-coil
    '''
    print os.getpid(), ' :started basic calc'
    for tmp in range(1,len(coil_points)):
        dl = coil_points[tmp] - coil_points[tmp-1]
        dl_loc = (coil_points[tmp] + coil_points[tmp-1])/2.
        points = np.transpose(np.array([xyz_X.flatten() - dl_loc[0], xyz_Y.flatten() - dl_loc[1], xyz_Z.flatten() - dl_loc[2]]))
        currents = np.transpose(np.array([xyz_X.flatten()*0+dl[0], xyz_Y.flatten()*0+dl[1], xyz_Z.flatten()*0+dl[2]]))*coil_currents
        mag_r = np.sqrt(points[:,0]**2+points[:,1]**2+points[:,2]**2)
        mag_r = np.transpose(np.array([mag_r,mag_r,mag_r]))
        r_hat = points/mag_r
        low_values_indices = mag_r < dist_thresh  # Where too close to I-coil
        mag_r[low_values_indices] = dist_thresh  # Replace with min dist_thresh
        if tmp==1:
            B = np.cross(currents, r_hat)*10**(-3)/(mag_r**2) #10^-3 is from G/kA and mu_0
        else:
            B += np.cross(currents, r_hat)*10**(-3)/(mag_r**2) #10^-3 is from G/kA and mu_0
    print os.getpid(), ' :finished basic calc'    
    return B



def individual_biot_calc_multi(pickle_name, coil_point_list, xyz_X, xyz_Y, xyz_Z, coil_currents, dist_thresh, i):
    '''
    In charge of running an individual job, and pickling the result
    This is to allow multiprocessing
    '''
    print os.getpid(), ' started'
    start_time = time.time()
    B = basic_calculation(coil_point_list[i], xyz_X, xyz_Y, xyz_Z, coil_currents[i], dist_thresh)
    tmp_pickle_file = file(pickle_name,'w')
    pickle.dump(B,tmp_pickle_file)
    tmp_pickle_file.close()
    print os.getpid(), ' pickle dumped, time for this coil : ', time.time() - start_time


def biot_calc(coil_point_list, xyz_X, xyz_Y, xyz_Z, coil_currents, dist_thresh=0., multi_proc=None):
    '''
    run the biot savart calculation for the i-coils described in coil_point_list
    can be run in multi processor mode or single processor mode
    Number of processes is controlled by multi_proc
    multi_proc sets how the multiprocessing is done
    '''

    start_time = time.time()
    Bx = xyz_X * 0; By = xyz_X * 0; Bz = xyz_X * 0
    mu_0 = 4.*np.pi*(10**(-7))
    constant = mu_0/(4.*np.pi)
    B_list = []
    if multi_proc ==None:
        for i in range(0,len(coil_point_list)):
            print 'in single proc option'
            #coil_points = np.array(coil_point_list[i])
            B = basic_calculation(coil_point_list[i], xyz_X, xyz_Y, xyz_Z, coil_currents[i], dist_thresh)
            print 'time so far : ', time.time() - start_time
    elif multi_proc==3:
        pool_size = multiprocessing.cpu_count()
        pool = multiprocessing.Pool(processes=pool_size)
        #xyz_X_list = []; xyz_Y_list = []; xyz_Z_list = []; dist_thresh_list=[]
        #for i in range(0,len(coil_point_list)):
        #    xyz_X_list.append(xyz_X); xyz_Y_list.append(xyz_Y); xyz_Z_list.append(xyz_Z)
        #    dist_thresh_list.append(dist_thresh)
        print 'creating pool map'
        B_list = pool.map(basic_calculation_wrapper, itertools.izip(coil_point_list, itertools.repeat(xyz_X), 
                                                     itertools.repeat(xyz_Y), itertools.repeat(xyz_Z), 
                                                     coil_currents, itertools.repeat(dist_thresh)))
        print B_list
        print len(B_list)
        #B_list = pool.map(basic_calculation, coil_point_list, xyz_X_list, xyz_Y_list, xyz_Z_list, coil_currents, dist_thresh_list)
        #B_list = map(basic_calculation, coil_point_list, xyz_X_list, xyz_Y_list, xyz_Z_list, coil_currents, dist_thresh_list)
        print 'closing pool'
        pool.close() # no more tasks
        print 'waiting for pool to finish'
        pool.join()  # wrap up current tasks
        print 'pool finished'
        #basic_calculation(coil_points, xyz_X, xyz_Y, xyz_Z, coil_currents, dist_thresh)
    elif multi_proc==1:
        for i in range(0,len(coil_point_list)):
            print 'in single proc option'
            #coil_points = np.array(coil_point_list[i])
            job1 = biot_sav_job(coil_point_list[i], xyz_X, xyz_Y, xyz_Z, coil_currents[i], dist_thresh)
            B_list.append(job1.run_calc())
            #B = basic_calculation(coil_point_list[i], xyz_X, xyz_Y, xyz_Z, coil_currents[i], dist_thresh)
            print 'time so far : ', time.time() - start_time
    elif multi_proc==2:
        tasks = multiprocessing.JoinableQueue()
        results = multiprocessing.Queue()
        consumers = []
        num_consumers = multiprocessing.cpu_count()
        print 'Creating %d consumers' % num_consumers

        consumers = [multiprocessing.Process(target=biot_sav_worker, args=(tasks, results)) for tmp_i in range(0,num_consumers)]
        print 'Created workers ', len(consumers)

        for tmp_i in range(0,num_consumers):
            consumers[tmp_i].start()

        #consumers = [ biot_sav_worker(tasks, results)
        #              for i in xrange(num_consumers) ]

        #put all the jobs into the queue
        for i in range(0, len(coil_point_list)):
            tasks.put(biot_sav_job(coil_point_list[i], xyz_X, xyz_Y, xyz_Z, coil_currents[i], dist_thresh))

        #put a kill pill for each worker into the queue
        for i in range(0, num_consumers):
            tasks.put(None)

        while tasks.qsize() >= 1:
            print '... ', tasks.qsize(), results.qsize()
            time.sleep(2)
        # Wait for all of the tasks to finish
        print 'waiting for tasks to finish'
        tasks.join()

        print 'Put together the results'
        for i in range(0, len(coil_point_list)):
            result = results.get()
            B_list.append(result)


    else:
        workers = multi_proc
        print 'multi_proc, workers : ', workers
        tmp_filename_list = []; list_process = []

        #create the list of jobs
        for tmp_i in range(0,len(coil_point_list)):
            tmp_filename_list.append('tmp' + str(tmp_i))
            list_process.append(multiprocessing.Process(target=individual_biot_calc_multi,args=(tmp_filename_list[-1], coil_point_list, xyz_X, xyz_Y, xyz_Z, coil_currents, dist_thresh, tmp_i)))

        batch_list = range(0,len(coil_point_list), workers)
        print 'jobs created', batch_list

        #dispatch as many workers, and loop until all jobs are done
        for tmp in batch_list:
            print 'batch list ', tmp
            print range(tmp, min(tmp + workers, len(coil_point_list)))
            for tmp_i in range(tmp, min(tmp + workers, len(coil_point_list))):
                list_process[tmp_i].start()
            for tmp_i in range(tmp, min(tmp + workers, len(coil_point_list))):
                print 'joining jobs ', tmp_i
                list_process[tmp_i].join()
        print 'finished all jobs, starting to get results'

        #superpose all the results
        B_list = stich_together(tmp_filename_list)

    #Bx, By, Bz from the way it is stored in the B array
    for tmp_i in range(0,len(B_list)):
        B = B_list[tmp_i]
        Bx=Bx + B[:,0].reshape(xyz_X.shape)
        By=By + B[:,1].reshape(xyz_X.shape)
        Bz=Bz + B[:,2].reshape(xyz_X.shape)
    return Bx, By, Bz


def stich_together(list_files):
    '''
    Stich together the results from the individual pickle files
    Need to find a better way of doing this - possibly with a queue
    '''

    print 'Reading in the files and outputting the master'
    B_list = []
    for i in range(0,len(list_files)):
        print i,
        tmp_file = file(list_files[i], 'r')
        B_list.append(pickle.load(open(list_files[i])))
        tmp_file.close()
        os.system('rm '+ list_files[i])
    print ' finished stiching together'    
    return B_list

def generate_coil_points2(r, z, n, phi_zero, phi_range, axis):
    '''
    generate the geometry for an I-coil based on 4 points
    '''
    phi_zero = phi_zero/180.*np.pi
    phi_range = phi_range/180.*np.pi
    point0 = np.array([r[0]*np.cos(phi_zero-phi_range*0.5),r[0]*np.sin(phi_zero-phi_range*0.5),z[0]])
    point1 = np.array([r[1]*np.cos(phi_zero-phi_range*0.5),r[1]*np.sin(phi_zero-phi_range*0.5),z[1]])
    point2 = np.array([r[1]*np.cos(phi_zero+phi_range*0.5),r[1]*np.sin(phi_zero+phi_range*0.5),z[1]])
    point3 = np.array([r[0]*np.cos(phi_zero+phi_range*0.5),r[0]*np.sin(phi_zero+phi_range*0.5),z[0]])
    point_list = [point0, point1, point2, point3]
    #for i in range(0,len(point_list)):
    #    axis.text3D(point_list[i][0], point_list[i][1], point_list[i][2], str(i))
    coil_points = []
    for tmp in range(0,len(point_list)):
        start_point = point_list[tmp-1]
        end_point = point_list[tmp]
        #maybe try doing this in different co-ord system?
        if tmp==1 or tmp==3:
            increment = (end_point - start_point)/n
            for i in range(0,int(n)+1):
                coil_points.append(increment*i + start_point)
        else:
            phi_increment = phi_range/n
            for i in range(1,int(n)+1):
                if tmp==2:
                    coil_points.append(np.array([r[1]*np.cos(phi_zero-phi_range*0.5+phi_increment*(i)),r[1]*np.sin(phi_zero-phi_range*0.5+phi_increment*(i)),z[1]]))
                else:
                    coil_points.append(np.array([r[0]*np.cos(phi_zero+phi_range*0.5-phi_increment*(i-1)),r[0]*np.sin(phi_zero+phi_range*0.5-phi_increment*(i-1)),z[0]]))
                    pass
                    #coil_points.append(np.array([r[0]*np.cos(phi_zero-phi_range+phi_increment*i),r[0]*np.sin(phi_zero-phi_range+phi_increment*i),z[0]]))
    coil_points.append(coil_points[0])
    return coil_points


def generate_pickupcoil_points(probe, probe_type, Rprobe, Zprobe, tprobe, lprobe, phi_loc, phi_width, n_phi_width= 30, Navg=30):
    '''
    generate the geometry for an I-coil based on 4 points
    '''
    Nprobe = len(probe)
    xyz_X_list = []; xyz_Y_list = []; xyz_Z_list=[]; phi_array_list = []

    for k in range(0, Nprobe):
        if probe_type[k] == 1:
            Rprobek=Rprobe[k] + lprobe[k]/2.*np.cos(tprobe[k])*np.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] + lprobe[k]/2.*np.sin(tprobe[k])*np.linspace(-1,1,num = Navg)
        else:
            Rprobek=Rprobe[k] + lprobe[k]/2.*np.sin(tprobe[k])*np.linspace(-1,1,num = Navg)
            Zprobek=Zprobe[k] - lprobe[k]/2.*np.cos(tprobe[k])*np.linspace(-1,1,num = Navg)

        phi_zero = phi_loc[k]/180.*np.pi
        phi_range = phi_width[k]/180.*np.pi
        phi_array = np.tile(np.linspace(phi_zero-phi_range/2, phi_zero+phi_range/2, n_phi_width),(Navg,1))
        phi_array_list.append(phi_array)
        Rprobek_array = np.tile(Rprobek,(n_phi_width,1)).transpose()
        Zprobek_array = np.tile(Zprobek,(n_phi_width,1)).transpose()

        print 'array_shapes', phi_array.shape, Rprobek_array.shape, Zprobek_array.shape
        
        xyz_X_list.append(Rprobek_array*np.cos(phi_array))
        xyz_Y_list.append(Rprobek_array*np.sin(phi_array))
        xyz_Z_list.append(Zprobek_array)
        #print 'output_shapes',xyz_X_list.shape, xyz_Y_list.shape, xyz_Z_list.shape
    return xyz_X_list, xyz_Y_list, xyz_Z_list, phi_array_list


def I_coil_points(r_upper, z_upper, r_lower, z_lower, n_I_coil, phi_range, phi_zero, ax = None):
    coil_point_list = []
    for i, tmp in enumerate(phi_zero):
        #coil_points1 = generate_coil_points(r, z, n, tmp, phi_range)
        coil_points_up = generate_coil_points2(r_upper, z_upper, n_I_coil, tmp, phi_range[i], ax)
        coil_points_lower = generate_coil_points2(r_lower, z_lower, n_I_coil, tmp, phi_range[i], ax)
        tmp_array_up = np.array(coil_points_up)
        tmp_array_lower = np.array(coil_points_lower)
        if ax != None:
            ax.scatter3D(tmp_array_up[:,0],tmp_array_up[:,1],tmp_array_up[:,2])
            ax.scatter3D(tmp_array_lower[:,0],tmp_array_lower[:,1],tmp_array_lower[:,2])
            ax.plot(tmp_array_up[:,0],tmp_array_up[:,1],tmp_array_up[:,2])
            ax.plot(tmp_array_lower[:,0],tmp_array_lower[:,1],tmp_array_lower[:,2])
            for i in range(0, tmp_array_up.shape[0]):
                pass
                #ax.text3D(tmp_array_up[i,0],tmp_array_up[i,1],tmp_array_up[i,2], str(i), fontsize=8)
                #ax.text3D(tmp_array_lower[i,0],tmp_array_lower[i,1],tmp_array_lower[i,2], str(i),fontsize=8)
        coil_point_list.append(coil_points_up)
        coil_point_list.append(coil_points_lower)
    return coil_point_list
