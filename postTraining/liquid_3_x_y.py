import argparse
from datetime import datetime
import os
from tqdm import trange
import numpy as np
from PIL import Image
import gc
try:
	from manta import *
except ImportError:
	pass

parser = argparse.ArgumentParser()

# CHANGE many arguments please cross reference with liquid3_d_r
parser.add_argument("--log_dir", type=str, default='log_dir')
parser.add_argument("--num_param", type=int, default=3)
parser.add_argument("--path_format", type=str, default='%d_%d_%d.npz')
parser.add_argument("--p0", type=str, default='x')
parser.add_argument("--p1", type=str, default='y')
parser.add_argument("--p2", type=str, default='frames')

parser.add_argument("--min_x", type=float, default=0.3)
parser.add_argument("--max_x", type=float, default=0.5)
parser.add_argument("--num_x", type=int, default=10)
parser.add_argument("--min_y", type=float, default=0.5)
parser.add_argument("--max_y", type=float, default=0.7) # 180.0/n*(n-1), 5: 144, 10: 162
parser.add_argument("--num_y", type=int, default=10)
# parser.add_argument("--src_y_pos", type=float, default=0.6)
# parser.add_argument("--src_radius", type=float, default=0.1)
# parser.add_argument("--basin_y_pos", type=float, default=0.2)
parser.add_argument("--min_frames", type=int, default=0)
parser.add_argument("--max_frames", type=int, default=199)
parser.add_argument("--num_frames", type=int, default=200)
# parser.add_argument("--num_simulations", type=int, default=7500)

parser.add_argument("--resolution_x", type=int, default=256)
parser.add_argument("--resolution_y", type=int, default=128)
parser.add_argument("--resolution_z", type=int, default=1)
# parser.add_argument("--gravity", type=float, default=-1e-3)
parser.add_argument("--radius_factor", type=float, default=1)
parser.add_argument("--min_particles", type=int, default=3)
parser.add_argument("--bWidth", type=int, default=1)
parser.add_argument("--open_bound", type=bool, default=False)
parser.add_argument("--time_step", type=float, default=0.8)

args = parser.parse_args()

# CHANGE INIT
# INIT for almost everything
factor=75
resx=256
resy=128
res_x = int(factor*3.2)
res_y = int(factor*2.0)
radius_factor = 1.0
min_particles = 3
frames = 500
# res_z = 1
gs = vec3(resx, resy, 1)
#buoyancy = vec3(0,-4e-3 , 0)
gravity = vec3(0,-1e-3,0)
# CHANGE took width to global
width = int((resx-res_x)*0.5)


def advect():# CHANGE min_dist -> min_x and so on, similarly min_rot -> min_y and so on
	def get_param(p1, p2):
	    min_p1 = args.min_x
	    max_p1 = args.max_x
	    num_p1 = args.num_x
	    min_p2 = args.min_y
	    max_p2 = args.max_y
	    num_p2 = args.num_y
	    p1_ = p1/(num_p1-1) * (max_p1-min_p1) + min_p1
	    p2_ = p2/(num_p2-1) * (max_p2-min_p2) + min_p2
	    return p1_, p2_

	# p1, p2 = 2, 4
	p1, p2 = 5, 5
	p1_, p2_ = get_param(p1, p2)

	# CHANGE add INIT
	# INIT for fluidbasin
	basin_x_pos = p1_
	basin_y_pos = p2_


	v_path = os.path.join(args.log_dir, 'v')
	img_dir = os.path.join(args.log_dir, 'l_adv')
	if not os.path.exists(img_dir):
	    os.makedirs(img_dir)        
	# obj_dir = os.path.join(args.log_dir, 'obj_adv')
	# if not os.path.exists(obj_dir):
	#     os.makedirs(obj_dir)

	# solver params
	# CHANGE comment out res_x,res_y,res_z lines
	# FIXME comment the next three lines, and run again for all data So that fluid basin is considered w.r.t the factor
	res_x = args.resolution_x
	res_y = args.resolution_y
	res_z = args.resolution_z
	# CHANGE res_x ->resx,res_y->resy, res_z -> 1
	gs = vec3(resx, resy, 1)

	# CHANGE dim=2
	s = Solver(name='main', gridSize=gs, dim=2)
	s.timestep = args.time_step

	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	pp = s.create(BasicParticleSystem) 
	mesh = s.create(Mesh)

	# acceleration data for particle nbs
	pindex = s.create(ParticleIndexSystem) 
	gpi = s.create(IntGrid)

	# CHANGE args.bWidth -> width
	flags.initDomain(boundaryWidth=width)
	vel.clear()
	pp.clear()
	pindex.clear()
	gpi.clear()


	# CHANGE Add fluid basin
	# TODO Add fluidbasin here
	fluidBasin = Box(parent=s, p0=vec3(width,width,0), p1=vec3(res_x*basin_x_pos+width,res_y*basin_y_pos+width,1)) # basin

	# CHANGE lots of commenting out
	# fluidBasin = Box(parent=s, p0=gs*vec3(0,0,0), p1=gs*vec3(1.0,args.basin_y_pos,1.0)) # basin
	# fluidCenter = gs*vec3(0.5,args.src_y_pos,0.5)
	# r = gs.x*p1_
	# th1 = p2_/180.0*np.pi
	# th2 = th1 + np.pi
	# c1 = fluidCenter + vec3(r*np.cos(th1),0,r*np.sin(th1))
	# c2 = fluidCenter + vec3(r*np.cos(th2),0,r*np.sin(th2))
	# print(p1_, p2_, c1, c2)
		
	# fluidDrop1 = Sphere(parent=s, center=c1, radius=gs.x*args.src_radius)
	# fluidDrop2 = Sphere(parent=s, center=c2, radius=gs.x*args.src_radius)

	phi = fluidBasin.computeLevelset()
	# phi.join(fluidDrop1.computeLevelset())
	# phi.join(fluidDrop2.computeLevelset())

	flags.updateFromLevelset(phi)
	sampleLevelsetWithParticles(phi=phi, flags=flags, parts=pp, discretization=2, randomness=0.05)

	if (GUI):
	    gui = Gui()
	    gui.show(True)
	    gui.nextVec3Display()
	    gui.nextVec3Display()
	    gui.nextVec3Display()
	    gui.pause()
	# CHANGE res_x ->resx,res_y->resy, res_z -> 1
	l_ = np.zeros([1,resy,resx], dtype=np.float32)
	v = np.zeros([resy,resx,3], dtype=np.float32)
	for t in trange(args.num_frames):
	    v_path_ = os.path.join(v_path, str(t)+".npz" )# args.path_format % (p1, p2, t))
	    with np.load(v_path_) as data:
	        v[...,:2] = data['x']

	    copyArrayToGridMAC(v, vel)

	    markFluidCells(parts=pp, flags=flags)
	    # if i > 100:
	    # checkHang(parts=pp, vel=vel, flags=flags, threshold=0.01) # 0.05
	    extrapolateMACSimple(flags=flags, vel=vel, distance=4)

	    gridParticleIndex(parts=pp, flags=flags, indexSys=pindex, index=gpi)
	    # unionParticleLevelset(pp, pindex, flags, gpi, phi, args.radius_factor)
	    averagedParticleLevelset(pp, pindex, flags, gpi, phi, args.radius_factor, 1, 1)
	    phi.setBound(1, boundaryWidth=args.bWidth)
	    resetOutflow(flags=flags, parts=pp, index=gpi, indexSys=pindex)

	    # copy levelset to visualize
	    # CHANGE levelset saving np.mean(l_[:,::-1], axis=0) -> l_[::-1,:] 
	    copyGridToArrayLevelset(phi, l_)
	    l_file_path = os.path.join(img_dir, '%04d.png' % t)
	    l_img = np.mean(l_[:,::-1], axis=0)*255 # yx
	    l_img = np.stack((l_img,l_img,l_img), axis=-1).astype(np.uint8)
	    l_img = Image.fromarray(l_img)
	    print("savint to "+ l_file_path)
	    l_img.save(l_file_path)

	    # extrapolate levelset, needed by particle resampling in adjustNumber / resample
	    extrapolateLsSimple(phi=phi, distance=4, inside=True)

	    # set source grids for resampling, used in adjustNumber!
	    # CHANGE commented adjust Number
	    # adjustNumber(parts=pp, vel=vel, flags=flags, minParticles=args.min_particles,
	                 # maxParticles=2*args.min_particles, phi=phi, radiusFactor=args.radius_factor)

	    
	    pp.advectInGrid(flags=flags, vel=vel, integrationMode=IntRK4, deleteInObstacle=False)
	    
	    # # create mesh for vis.
	    # phi.createMesh(mesh)
	    # for iters in range(5):
	    #     smoothMesh(mesh=mesh, strength=1e-3, steps=10) 
	    #     subdivideMesh(mesh=mesh, minAngle=0.01, minLength=0.5, maxLength=3*0.5, cutTubes=True)
	    
	    # obj_path = os.path.join(obj_dir, '%04d.obj' % t)
	    # mesh.save(obj_path)
	    
	    # pt_path = os.path.join(pt_dir, '%04d.uni' % t)
	    # pp.save(pt_path)

	    s.step()

def main1():
	if not os.path.exists(args.log_dir):
	    os.makedirs(args.log_dir)

	field_type = ['v'] #, 'obj', 'pt', 'p', 's']
	for field in field_type:
	    field_path = os.path.join(args.log_dir,field)
	    if not os.path.exists(field_path):
	        os.makedirs(field_path)

	args_file = os.path.join(args.log_dir, 'args.txt')
	with open(args_file, 'w') as f:
	    print('%s: arguments' % datetime.now())
	    for k, v in vars(args).items():
	        print('  %s: %s' % (k, v))
	        f.write('%s: %s\n' % (k, v))
	# uncomment this line to cause an error 
	# import ssdajsj
	p1_space = np.linspace(args.min_x, 
	                        args.max_x,
	                        args.num_x)
	p2_space = np.linspace(args.min_y,
	                        args.max_y,
	                        args.num_y)
	p_list = np.array(np.meshgrid(p1_space, p2_space)).T.reshape(-1, 2)
	pi1_space = range(args.num_x)
	pi2_space = range(args.num_y)
	pi_list = np.array(np.meshgrid(pi1_space, pi2_space)).T.reshape(-1, 2)

	# FIXME comment the next three lines, and run again for all data
	res_x = args.resolution_x
	res_y = args.resolution_y
	res_z = args.resolution_z

	v_ = np.zeros([res_y,res_x,3], dtype=np.float32)
	# l_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
	# p_ = np.zeros([res_z,res_y,res_x], dtype=np.float32)
	# s_ = np.zeros([res_z,res_y,res_x,3], dtype=np.float32)

	v_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# l_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# p_range = [np.finfo(np.float).max, np.finfo(np.float).min]
	# s_range = [np.finfo(np.float).max, np.finfo(np.float).min]

	# solver params
	# CHANGE comment out res_x,res_y,res_z lines
	# res_x = args.resolution_x
	# res_y = args.resolution_y
	# res_z = args.resolution_z
	# CHANGE res_x ->resx,res_y->resy, res_z -> 1
	gs = vec3(resx, resy, 1)
	# gravity = vec3(0,args.gravity,0)

	# CHANGE dim=2
	s = Solver(name='main', gridSize=gs, dim=2)
	s.timestep = args.time_step

	flags = s.create(FlagGrid)
	vel = s.create(MACGrid)
	pressure = s.create(RealGrid)

	# flip
	velOld = s.create(MACGrid)
	tmpVec3 = s.create(VecGrid)

	pp = s.create(BasicParticleSystem) 
	pVel = pp.create(PdataVec3)
	# mesh = s.create(Mesh)

	# acceleration data for particle nbs
	pindex = s.create(ParticleIndexSystem) 
	gpi = s.create(IntGrid)

	# omega = s.create(VecGrid)
	# stream = s.create(VecGrid)
	# # vel_out = s.create(MACGrid)

	if (GUI):
	    gui = Gui()
	    gui.show(True)
	    gui.nextVec3Display()
	    gui.nextVec3Display()
	    gui.nextVec3Display()
	    #gui.pause()

	print('start generation')
	for i in trange(len(p_list), desc='scenes'):
		# CHANGE args.bWidth -> width
		flags.initDomain(boundaryWidth=width)

		vel.clear()
		pressure.clear()
		# stream.clear()

		velOld.clear()
		tmpVec3.clear()

		pp.clear()
		pVel.clear()

		# scene setup
		# TODO Add fluidbasin here
		# fluidBasin = Box(parent=s, p0=gs*vec3(0,0,0), p1=gs*vec3(1.0,args.basin_y_pos,1.0)) # basin
		# fluidCenter = gs*vec3(0.5,args.src_y_pos,0.5)

		p0, p1 = p_list[i][0], p_list[i][1]
		# CHANGE add INIT
		# INIT for fluidbasin
		basin_x_pos = p0
		basin_y_pos = p1

		# CHANGE Add fluid basin
		# TODO Add fluidbasin here
		fluidBasin = Box(parent=s, p0=vec3(width,width,0), p1=vec3(res_x*basin_x_pos+width,res_y*basin_y_pos+width,1)) # basin

		# CHANGE lots of commenting out
		# r = gs.x*p0
		# th1 = p1/180.0*np.pi
		# th2 = th1 + np.pi
		# c1 = fluidCenter + vec3(r*np.cos(th1),0,r*np.sin(th1))
		# c2 = fluidCenter + vec3(r*np.cos(th2),0,r*np.sin(th2))
		# print(p0, p1, c1, c2)

		# fluidDrop1 = Sphere(parent=s, center=c1, radius=gs.x*args.src_radius)
		# fluidDrop2 = Sphere(parent=s, center=c2, radius=gs.x*args.src_radius)

		# fluidVel1 = Sphere(parent=s, center=c1, radius=gs.x*(args.src_radius+0.05))
		# fluidVel2 = Sphere(parent=s, center=c2, radius=gs.x*(args.src_radius+0.05))
		# fluidSetVel = vec3(0,-1,0)

		phi = fluidBasin.computeLevelset()
		# phi.join(fluidDrop1.computeLevelset())
		# phi.join(fluidDrop2.computeLevelset())

		flags.updateFromLevelset(phi)
		sampleLevelsetWithParticles(phi=phi, flags=flags, parts=pp, discretization=2, randomness=0.05)

		# set initial velocity
		# fluidVel1.applyToGrid(grid=vel, value=fluidSetVel)
		# fluidVel2.applyToGrid(grid=vel, value=fluidSetVel)
		mapGridToPartsVec3(source=vel, parts=pp, target=pVel)

		# CHANGE res_x ->resx,res_y->resy, res_z -> 1
		# INIT , which coincidently is also CHANGE
		l_ = np.zeros([1,resy,resx], dtype=np.float32)

		img_dir = field_path+"imgs_l_big/"
		if not os.path.exists(img_dir):
			os.makedirs(img_dir)
		for t in trange(args.num_frames, desc='sim'):
			# FLIP 
			pp.advectInGrid(flags=flags, vel=vel, integrationMode=IntRK4, deleteInObstacle=False)

			# make sure we have velocities throught liquid region
			mapPartsToMAC(vel=vel, flags=flags, velOld=velOld, parts=pp, partVel=pVel, weight=tmpVec3) 
			extrapolateMACFromWeight(vel=vel, distance=2, weight=tmpVec3)  # note, tmpVec3 could be free'd now...
			markFluidCells(parts=pp, flags=flags)

			# create approximate surface level set, resample particles
			gridParticleIndex(parts=pp , flags=flags, indexSys=pindex, index=gpi)
			# unionParticleLevelset(pp, pindex, flags, gpi, phi, args.radius_factor)
			averagedParticleLevelset(pp, pindex, flags, gpi, phi, args.radius_factor, 1, 1)
			phi.setBound(1, boundaryWidth=args.bWidth)
			resetOutflow(flags=flags, parts=pp, index=gpi, indexSys=pindex) 

			# copyGridToArrayLevelset(phi, l_)
			# CHANGE add levelset saving
			
			copyGridToArrayLevelset(phi, l_)
			l_file_path = os.path.join(img_dir, '%04d.png' % t)
			l_img = np.mean(l_[:,::-1], axis=0)*255 # yx
			l_img = np.stack((l_img,l_img,l_img), axis=-1).astype(np.uint8)
			l_img = Image.fromarray(l_img)
			l_img.save(l_file_path)

			# extend levelset somewhat, needed by particle resampling in adjustNumber
			extrapolateLsSimple(phi=phi, distance=4, inside=True)

			# forces & pressure solve
			addGravity(flags=flags, vel=vel, gravity=gravity)
			setWallBcs(flags=flags, vel=vel)	
			solvePressure(flags=flags, vel=vel, pressure=pressure, phi=phi)
			setWallBcs(flags=flags, vel=vel)
			# copyGridToArrayReal(pressure, p_)

			# set source grids for resampling, used in adjustNumber!
			pVel.setSource(vel, isMAC=True)
			# CHANGE commented adjust Number
			# adjustNumber(parts=pp, vel=vel, flags=flags, minParticles=args.min_particles, maxParticles=2*args.min_particles, phi=phi, radiusFactor=args.radius_factor)

			# make sure we have proper velocities
			extrapolateMACSimple(flags=flags, vel=vel)
			flipVelocityUpdate(vel=vel, velOld=velOld, flags=flags, parts=pp, partVel=pVel, flipRatio=0.97)
			copyGridToArrayMAC(vel, v_)

			# # create mesh for vis.
			# phi.createMesh(mesh)
			# for iters in range(5):
			# 	smoothMesh(mesh=mesh, strength=1e-3, steps=10) 
			# 	subdivideMesh(mesh=mesh, minAngle=0.01, minLength=0.5, maxLength=3*0.5, cutTubes=True)

			# getStreamfunction(flags=flags, vel=vel, grid=stream)
			# copyGridToArrayReal(stream, s_)

			v_range = [np.minimum(v_range[0], v_.min()),
			            np.maximum(v_range[1], v_.max())]
			# l_range = [np.minimum(l_range[0], l_.min()),
			# 		   np.maximum(l_range[1], l_.max())]
			# p_range = [np.minimum(p_range[0], p_.min()),
			# 		   np.maximum(p_range[1], p_.max())]
			# s_range = [np.minimum(s_range[0], s_.min()),
			# 		   np.maximum(s_range[1], s_.max())]

			param_ = [p0, p1, t]
			pit = tuple(pi_list[i].tolist() + [t])

			v_file_path = os.path.join(args.log_dir, 'v', args.path_format % pit)
			np.savez_compressed(v_file_path, 
			                    x=v_[...,:2],
			                    y=param_)

			# # save particles
			# pt_file_path = os.path.join(args.log_dir, 'pt', '%d_%d_%d.uni' % pit)
			# pp.save(pt_file_path)

			# l_file_path = os.path.join(args.log_dir, 'l', args.path_format % pit)
			# np.savez_compressed(l_file_path, 
			# 					x=np.expand_dims(l_, axis=-1),
			# 					y=param_)

			# obj_file_path = os.path.join(args.log_dir, 'obj', '%.2e_%.2e_%d.obj' % tuple(param_))
			# mesh.save(obj_file_path)

			# p_file_path = os.path.join(args.log_dir, 'p', args.path_format % pit)
			# np.savez_compressed(p_file_path, 
			# 					x=np.expand_dims(p_, axis=-1),
			# 					y=param_)

			# s_file_path = os.path.join(args.log_dir, 's', args.path_format % pit)
			# np.savez_compressed(s_file_path, 
			# 					x=s_, # yxzd
			# 					y=param_)

			s.step()
		gc.collect()	

	vrange_file = os.path.join(args.log_dir, 'v_range.txt')
	with open(vrange_file, 'w') as f:
		print('%s: velocity min %.3f max %.3f' % (datetime.now(), v_range[0], v_range[1]))
		f.write('%.3f\n' % v_range[0])
		f.write('%.3f' % v_range[1])

	# lrange_file = os.path.join(args.log_dir, 'l_range.txt')
	# with open(lrange_file, 'w') as f:
	#     print('%s: levelset min %.3f max %.3f' % (datetime.now(), l_range[0], l_range[1]))
	#     f.write('%.3f\n' % l_range[0])
	#     f.write('%.3f' % l_range[1])

	# prange_file = os.path.join(args.log_dir, 'p_range.txt')
	# with open(prange_file, 'w') as f:
	# 	print('%s: pressure min %.3f max %.3f' % (datetime.now(), p_range[0], p_range[1]))
	# 	f.write('%.3f\n' % p_range[0])
	# 	f.write('%.3f' % p_range[1])

	# srange_file = os.path.join(args.log_dir, 's_range.txt')
	# with open(srange_file, 'w') as f:
	# 	print('%s: stream min %.3f max %.3f' % (datetime.now(), s_range[0], s_range[1]))
	# 	f.write('%.3f\n' % s_range[0])
	# 	f.write('%.3f' % s_range[1])

	print('Done')


if __name__ == '__main__':
    # main()

    # advection test
    advect()