import numpy as np
from glob import glob
import matplotlib.dates as mdates
from datetime import datetime
from sunpy.map import Map
from sunpy.coordinates import frames
from astropy.coordinates import SkyCoord
from sunpy.physics.differential_rotation import solar_rotate_coordinate
import astropy.units as u
from scipy.interpolate import interp2d
import os



def get_aia_filename_list(aia_dir_path): 
    """ get a filelist containing the AIA data
    
    Args: 
        aia_dir_path: directory for saving the AIA data

    Returns: 
        aia_filename_list_sorted: sorted filelist. 

    """
    aia_filename_list = glob(aia_dir_path)
    aia_filename_list.sort(key = lambda x:  mdates.date2num(datetime.strptime( x[-36:-21] , '%Y%m%d_%H%M%S' )) )
    return aia_filename_list




def get_sharp_filename_list(sharp_dir_path): 
    """ get a filelist containing the SHARP data
    
    Args: 
        sharp_dir_path: directory for SHARP data

    Returns: 
        sharp_filename_list_sorted: sorted filelist. 

    """
    sharp_filename_list = glob(sharp_dir_path)
    sharp_filename_list.sort(key = lambda x:  mdates.date2num(datetime.strptime( x[-27:-12] , '%Y%m%d_%H%M%S' )) )

    return sharp_filename_list



def find_close_sharpmap(aia_filename, sharp_filename_list_sorted): 
    """ find the sharp data that is closet to the time of the AIA data from a list

    Args: 
        aia_filename: the target aia map 
        sharp_filename_list_sorted: a list of sharp map covering the necessary time 
                                    range for transforming.

    Returns: 
        sharp_filename: the index number of  sharp_filename_list_sorted, which is the frame that
                    is most close to the time of AIA time. 

    """
    sharp_time_num_list = np.array([ mdates.date2num(
                            datetime.strptime( x[-27:-12] , '%Y%m%d_%H%M%S' ))
                            for x in sharp_filename_list_sorted ])

    aia_time_num = mdates.date2num(datetime.strptime( aia_filename[-36:-21] , '%Y%m%d_%H%M%S' ) ) 
    nearest_ind = np.argmin(np.abs(aia_time_num-sharp_time_num_list))
    sharp_filename = sharp_filename_list_sorted[nearest_ind]
    

    return sharp_filename





class Saarp_generator(): 
    """ transforming the projection of AIA data into CEA to coalign the SHARP data""" 

    def __init__( self, aiamap, sharpmap ): 
        """ transform the projection of AIA data and overlying on the corresponding 
            sharpdata (at close time)

        Args:
            aiamap: AIA map data. 
            sharpmap: sharp map data  
            
        Returns: 
            transformed AIA data (in npz file)
        """
        self.sharpmap = sharpmap
        self.aiamap = aiamap
        self.saarp_map = None
        self.lon_s = None
        self.lat_s = None


        


    def make_map(self, ): 
        """ Method: to generate transformed AIA map which is Coaligning with the SHARP data at the 
        taken at a close time. 

        Args: 
        
        return: 
            saarp_map: transformed AIA data (sunpy Map) 
        """   
        
        def get_fov_sharp(map): 
            """find the FOV of the SHARP map 

            Args: 
                sharpmap

            Returns: 
                fov_sharp: a 4-element array contains the fov information.
            """
            ind = map.meta
            lon1 = ind['londtmin']
            lon2 = ind['londtmax']
            lat1 = ind['latdtmin']
            lat2 = ind['latdtmax']

            fov_sharp = np.array([lon1, lon2, lat1, lat2]) 

            return fov_sharp
        
        def get_fov_aia(map): 
            """ find the fov pf the AIA map

            Args: 
                aiamap

            Returns: 
                fov_aia: a 4-element array contains the fov information. 
                
            """
            
            bottom_left_aia = map.bottom_left_coord
            top_right_aia = map.top_right_coord

            x0_aia = bottom_left_aia.Tx.value
            x1_aia = top_right_aia.Tx.value
            y0_aia = bottom_left_aia.Ty.value
            y1_aia = top_right_aia.Ty.value

            fov_aia = np.array([x0_aia, x1_aia, y0_aia, y1_aia])  

            return fov_aia


        def get_xy_array(map, fov): 
            """generate the x, y array of the map

            Args: 
                fov: a 4-element array recording x1, x2, y1, y2 

            Returns: 
                x: array representing the x-direction 
                y: array representing the y-direction  
            """
            ind = map.meta
            x1, x2, y1, y2 = fov[0], fov[1], fov[2], fov[3]
            x = np.linspace(x1, x2, ind['naxis1'] )
            y = np.linspace(y1, y2, ind['naxis2'])

            return x, y 


        def HS_2_HP(Time, lon_mesh, lat_mesh): 
            """transform the longitude array, latitude array to the Helio projective coordinate

            Args: 
                Time: observation time 
                lon_s, lat_s: longitude and latitude coordinate in Heliographic Stonyhurst coordinate
            
            Returns: 
                x_s: lon_s in CCD coordinate 
                y_s: lat_s in CCD coordinate
            """ 

            m, n = lon_mesh.shape
            
            # get the CCD coordinate of each pixel in SHARP data:
            sharp_skycoord = SkyCoord(  lon_mesh.flatten()*u.deg, lat_mesh.flatten()*u.deg, 
                                        frame=frames.HeliographicStonyhurst, obstime=Time, observer='earth' )
            sharp_skycoord_new = sharp_skycoord.transform_to(frames.Helioprojective) 
            Tx_flatten = sharp_skycoord_new.Tx.value
            Ty_flatten = sharp_skycoord_new.Ty.value

            Tx_mesh = Tx_flatten.reshape(m, n)
            Ty_mesh = Ty_flatten.reshape(m, n)

            return Tx_mesh, Ty_mesh
        
        
        def interpolate_to_saarp(aiamap, xx, yy):
            """interpolate the fullmap AIA to the sharp data meshgrid 

            Args: 
                aiamap: map you want to regrid (bascially a full map)
                xx: meshgrid X for interpolating to
                yy: meshgrid Y for interpolating to

            Returns: 
                saarp_data: saarp image

            """
            # apply 2D-interpolation
            # make axes of aiamap for making interpolation function
            aiadata = aiamap.data

            fov_aia = get_fov_aia(aiamap)
            aia_x, aia_y =  get_xy_array(aiamap, fov_aia)

            # make 2D nterpolation function: f
            f = interp2d(aia_x, aia_y, aiadata, kind='cubic')

            # blank array for storing the aia intensity of each pixel
            aia_new = xx.flatten().copy()*0

            # implement the interpolation; get the corresponding AIA 1600 intensity
            for i in range(xx.flatten().size):
                aia_new[i] = f(xx.flatten()[i], yy.flatten()[i])

            # reshape the data. (saarp: AIA 1600 observation transformed to CEA coordinate)
            m, n = xx.shape
            saarp_data = aia_new.reshape(m, n)
            return saarp_data

        
        # find the fov of sharp map
        fov_sharp = get_fov_sharp(self.sharpmap)

        # generate the longitude array and the latitude array (in Heliographic Stonyhurst coordinate)
        lon_s, lat_s = get_xy_array(self.sharpmap, fov_sharp )

        # # *************** DELETE BELOW AFTER TESTING ***************
        # lon_s = lon_s[100:300]
        # lat_s = lat_s[50:200]
        # # *************** DELETE ABOVE AFTER TESTING ***************

        # find the CCD coordinate of the  longitude array
        T_sharp = self.sharpmap.date
        meshgrid_sharp_lon, meshgrid_sharp_lat = np.meshgrid(lon_s, lat_s)  # prepare the meshgrid  
        xx, yy = HS_2_HP(T_sharp, meshgrid_sharp_lon, meshgrid_sharp_lat) 
    
        # interpolate to the saarp map 
        saarp_data = interpolate_to_saarp(self.aiamap, xx, yy)

        # variables will be required in the output
        self.saarp_data = saarp_data
        self.lon_s = lon_s
        self.lat_s = lat_s
        

        return saarp_data

    def save_saarp(self, filename, out_dir=""):
        """ save the saarp data into a npz file
        Args:
            filename: the name of he npz file
            out_dir: path for storing the file

        return: 
            output the data under  out_dir
        """ 
        if out_dir != "": 
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)


        if self.saarp_data is None: 
            print("please implement .make_map")
        else: 
            if filename is None: 
                print('filename is required.')
            else: 
                np.savez( out_dir+filename,
                    lon_s=self.lon_s, lat_s=self.lat_s, saarp_data=self.saarp_data,
                    aia_time=self.aiamap.date, sharp_time=self.sharpmap.date )
        
        




        

