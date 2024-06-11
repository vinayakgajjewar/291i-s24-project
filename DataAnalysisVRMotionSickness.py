# %%
import pandas as pd
import numpy as np
from numpy import polyfit
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix
from sklearn.manifold import TSNE

# %%
# Path to the data containing the motion sickness scores
data_path = '/Volumes/Expansion/20645-Swathi/MS/Academics/CS291I_S24/SelfReportData'
# Path to the dataset for all games
game_data_path = '/Volumes/Expansion/20645-Swathi/MS/Academics/CS291I_S24/Dataset'

# %%
all_dataset_files = []
# Get all file names by walking through game_data_path
for root, dirs, files in os.walk(game_data_path):
    for file in files:
        if file.startswith('.'): continue
        all_dataset_files.append(os.path.join(root, file))
# print(all_dataset_files)

# %%
def GetFrameNumberFromTime(csvfile, time):
    # get first part of game name from csv file
    game_name = csvfile.split(' ')[2]
    # get participant name from csv file
    participant = csvfile.split(' ')[1]
    # for control.csv in the folder corresponding to the game and participant, get the frame number corresponding to the time
    for file in all_dataset_files:
        if game_name in file and participant in file and 'control.csv' in file:
            control_data = pd.read_csv(file)
            # get the frame number corresponding to timestamp closest to the time
            times_col = control_data['timestamp']
            row_num = np.argmin(np.abs(times_col-time))
            frames_col = control_data['framecounter']
            frame_number = frames_col[row_num]
            return frame_number
    

# %%
def GetMotionSicknessScore(csvfile):
    scores = []
    frames = []
    timestamps = []
    df = pd.read_csv(csvfile, header=None)
    # CSV is of the form
    # column 1 - timestamp, column 2 - motion sickness score, column 3 - auto/manual
    # 1683190996512652, 1, auto
    # 1683191006719312, 4, auto
    # 1683191009509684, 1, manual
    # Add scores to a list
    for index, row in df.iterrows():
        scores.append(row.iloc[1])
        # get timestamp from raw epoch time
        timestamp_raw = row.iloc[0]
        # timestamp = pd.to_datetime(timestamp_raw, unit='us')
        # print("Score: ", row.iloc[1], " Timestamp: ", timestamp_raw/1000)
        # divide the raw timestamp by 1000 to get the timestamp in seconds
        # all the other csv files have timestamps in seconds
        frames.append(GetFrameNumberFromTime(csvfile, timestamp_raw/1000))
        timestamps.append(timestamp_raw/1000)
    print (scores)
    print (frames)
    return scores, frames, timestamps
        
        
    

# %% [markdown]
# ### Motion sickness scores over time

# %%
# plot graph of size 20*10 with each subgrah being 2*5
plt.figure(figsize=(20,10))

games = ['Beat Saber','Cartoon Network','Monster Awakens', 'Traffic Cop', 'Voxel Shot','VR Rome','Epic Roller Coasters','Mini Motor Racing', 'Pottery']
markers = ['o','s','^','*','x','v','+']
colors = ['r','g','b','c','m', 'y', 'k']

gameindex = 1
# for every game, plot the motion sickness scores for every participant id
# participant id is the first number in the filename
for game in games:
    participants = []
    all_scores = []  # List to store scores for all participants in this game
    all_frames = []  # List to store frames for all participants in this game
    plt.subplot(2,5,gameindex)
    i = 0
    for filename in os.listdir(data_path):
        if not filename.startswith(".") and filename.endswith(".csv") and game in filename:
            scores = []
            frames = []
            # Get the participant id
            participant_id = filename.split(' ')[0]
            participants.append(participant_id)
            print("Participant ID: ", participant_id, " Game: ", filename.split(' ')[2], " i: ",i, " gameindex: ", gameindex)
            scores, frames, times = GetMotionSicknessScore(data_path + '/' + filename)
            if (frames[0] != None):
                all_frames.extend(frames)
                all_scores.extend(scores)
            # plot the scores for each participant as a separate line in the same graph for the game
            plt.scatter(frames,scores,marker=markers[i],color=colors[i],label=participant_id)
            i += 1
        else:
            continue
    plt.title(game)
    plt.xlabel('Frame number')
    plt.ylabel('Motion Sickness Score')
    plt.legend(participants)

    print(all_frames)
    print(all_scores)
    if len(all_frames) > 0 and all_frames[0] != None: 
        z = np.polyfit(all_frames, all_scores, 1)
        p = np.poly1d(z)
        plt.plot(frames,p(frames),color=colors[i],label=game+' trendline')
    gameindex += 1

# %% [markdown]
# ### Light Intensity and Motion Sickness scores

# %%
def GetNumOfLightsPerFrame(file):
    df = pd.read_csv(file)
    frames = list(set(df['framecounter']))
    name_col = df['name']
    frame_col = df['framecounter']
    total_num_lights = [] # stores an array of nums where each element corresponds to number of lights in a single frame
    lights_in_frame = []
    for i in range(len(name_col)):
        if i == 0 or frame_col[i] == frame_col[i-1]:
            lights_in_frame.append(name_col[i])
        else:
            total_num_lights.append(len(lights_in_frame))
            lights_in_frame = []
            lights_in_frame.append(name_col[i])
        
    total_num_lights.append(len(lights_in_frame))
    print(len(total_num_lights))
    print("frames: ", frames)
    return total_num_lights, frames


# %%
def GetLightIntensity(file):
    # get first part of game name from csv file
    game_name = file.split(' ')[2]
    # get participant name from csv file
    participant = file.split(' ')[1]
    print("GetLightIntensity: game: ", game_name, " participant: ", participant, " filename: ", file) 
    for item in all_dataset_files:
        if game_name in item and participant in item and 'light.csv' in item:
            num_of_lights_per_frame, frames = GetNumOfLightsPerFrame(item)
            print("num of lights per frame: ", num_of_lights_per_frame)
            print("frames: ", frames)
            return num_of_lights_per_frame, frames

# %%
games = ['Beat Saber','Cartoon Network', 'Monster Awakens', 'Traffic Cop', 'Voxel Shot','VR Rome','Epic Roller Coasters','Mini Motor Racing', 'Pottery']
#games = ['Epic Roller Coasters', 'Cartoon Network'] # Plot only for this game since this is the only one having varying light intensity. The rest of the games have constant light intensity
markers = ['o','s','^','*','x','v','+']
colors = ['r','g','b','c','m', 'k', 'y']

game_index = 1

for game in games: 
      
    print(game)
    fig, ax = plt.subplots(2, 3, figsize=(25,10))
    fig.suptitle(game)
    
    participants = []
    all_scores = []  # List to store scores for all participants in this game
    all_frames = []  # List to store frames for all participants in this game
    all_lights = []  # List to store number of lights per frame for all participants in this game
    all_light_frames = [] # List to store frames for all lights in this game
    
    i = 0

    for filename in os.listdir(data_path):
        if not filename.startswith(".") and filename.endswith(".csv") and game in filename:

            scores = []
            frames = []
            # Get the participant id
            participant_id = filename.split(' ')[0]
            participants.append(participant_id)
            print("Participant ID: ", participant_id, " Game: ", filename.split(' ')[2], " i: ",i, " gameindex: ", game_index, " file: ", filename)
            
            scores, frames, times = GetMotionSicknessScore(data_path + '/' + filename)
            num_of_lights_per_frame, light_frames = GetLightIntensity(data_path + '/' + filename)
            
            if (len(frames) > 0 and  frames[0] != None and len(scores) > 0 and scores[0] != None 
            and len(num_of_lights_per_frame) > 0 and num_of_lights_per_frame[0] != None and len(light_frames) > 0 and light_frames[0] != None):
                
                ax[int(i/3)][i%3].scatter(frames,scores,marker=markers[i],color=colors[i],label=participant_id)
                ax[int(i/3)][i%3].set_ylabel("Motion Sickness score",color="green")
                ax[int(i/3)][i%3].set_xlabel('Frame Number')
                ax[int(i/3)][i%3].legend()
                
                ax2 = ax[int(i/3)][i%3].twinx()
                ax2.scatter(light_frames, num_of_lights_per_frame, marker='+', color='y')
                ax2.set_ylabel("Number of lights per frame",color="blue")
                
                if (len(frames) > 0 and frames[0] != None):
                    z = np.polyfit(frames, scores, 1)
                    p = np.poly1d(z)
                    ax[int(i/3)][i%3].plot(frames,p(frames))
                
                if (len(num_of_lights_per_frame) > 0 and num_of_lights_per_frame[0] != None):
                    z = np.polyfit(light_frames, num_of_lights_per_frame, 1)
                    p = np.poly1d(z)
                    ax2.plot(light_frames,p(light_frames),color='b')
            
            i += 1
        else:
            continue
    
    game_index += 1

fig.tight_layout()

# %% [markdown]
# ### Headset rotational movement and motion sickness

# %%
'''
We will get the angular velocity at 1s intervals since the plot becomes too cluttered with millisecond time
'''
def GetHeadsetAngularVelocity(file, getAllRows=False):
    
    game_name = file.split(' ')[2]
    # get participant name from csv file
    participant = file.split(' ')[1]
    print("GetHeadsetAngularVelocity: game: ", game_name, " participant: ", participant, " filename: ", file) 
    
    frames = []
    angular_vel = [[]]
    angular_velocity_x = []
    angular_velocity_y = []
    angular_velocity_z = []
    magnitudes = []
    timestamps = []
    
    for item in all_dataset_files:
        if game_name in item and participant in item and 'pose.csv' in item:
            df = pd.read_csv(item)
            # filter for device_id = 0 
            device_id_col = df['device_id']
            angular_velocity_col = df['angularVelocity']
            frame_counter_col = df['framecounter']
            timestamps_col = df['timestamp']
            # get frames where device_id = 0
            prev_timestamp = 0
            for i in range(len(device_id_col)):
                index = -1
                if device_id_col[i] == 0:
                    if i == 0 or getAllRows == True:
                        curr_timestamp = timestamps_col[i]
                        index = i 
                    else:
                        curr_timestamp = prev_timestamp + 20000
                        index = np.argmin(np.abs(timestamps_col-curr_timestamp))
                        
                    frames.append(frame_counter_col[index])
                    angular_vel_vector = angular_velocity_col[index].split(' ')
                    angular_velocity_x.append(float(angular_vel_vector[0]))
                    angular_velocity_y.append(float(angular_vel_vector[1]))
                    angular_velocity_z.append(float(angular_vel_vector[2]))
                    magnitudes.append(np.sqrt(float(angular_vel_vector[0])**2 + float(angular_vel_vector[1])**2 + float(angular_vel_vector[2])**2))
                    # add the angular velocity vector to the list
                    angular_vel.append([float(angular_vel_vector[0]), float(angular_vel_vector[1]), float(angular_vel_vector[2])])
                    timestamps.append(timestamps_col[index])
                    
                    prev_timestamp = curr_timestamp
            
    angular_vel.pop(0)
    
    print("Frames: ", frames)
    print("Angular velocity x: ", angular_velocity_x)
    print("Angular velocity y: ", angular_velocity_y)
    print("Angular velocity z: ", angular_velocity_z)
    print("Timestamps: ", timestamps)
    print("Angular vel: ", angular_vel)
    

    return frames, angular_velocity_x, angular_velocity_y, angular_velocity_z, magnitudes,angular_vel, timestamps
    

# %%
#games = ['Beat Saber','Cartoon Network', 'Monster Awakens', 'Traffic Cop', 'Voxel Shot','VR Rome','Epic Roller Coasters','Mini Motor Racing', 'Pottery']
games = ['Beat Saber']
markers = ['o','s','^','*','x','v','+', 'd', 'p', 'h']
colors = ['r','g','b','c','m', 'k', 'aquamarine','pink','orange','purple']

game_index = 1

for game in games:   
    print(game)
    participants = []
    all_scores = []  # List to store scores for all participants in this game
    all_frames = []  # List to store frames for all participants in this game
    
    fig, ax = plt.subplots(2,3,figsize=(25,10))
    fig.suptitle(game)
    
    i = 0
    
    for filename in os.listdir(data_path):
        if not filename.startswith(".") and filename.endswith(".csv") and game in filename:

            scores = []
            frames = []
            # Get the participant id
            participant_id = filename.split(' ')[0]
            participants.append(participant_id)
            print("Participant ID: ", participant_id, " Game: ", filename.split(' ')[2], " i: ",i, " gameindex: ", game_index, " file: ", filename)    
            
            scores, frames, times = GetMotionSicknessScore(data_path + '/' + filename)
            
            angular_vel_frames, angular_velocity_x, angular_velocity_y, angular_velocity_z, magnitudes, angular_vel, timestamps = GetHeadsetAngularVelocity(data_path + '/' + filename)
            
            ax[int(i/3)][i%3].scatter(times,scores,marker=markers[i],color=colors[i],label='Sickness score')
            z = np.polyfit(times, scores, 1)
            p = np.poly1d(z)
            ax[int(i/3)][i%3].plot(times,p(times),color=colors[i])
            ax[int(i/3)][i%3].set_ylabel("Motion Sickness score",color="green")
            
            ax2 = ax[int(i/3)][i%3].twinx()
            ax2.scatter(timestamps, angular_velocity_x, marker='o', color='lightblue', label='Angular Velocity X')
            ax2.scatter(timestamps, angular_velocity_y, marker='+', color='lightgray', label='Angular Velocity Y')
            ax2.scatter(timestamps, angular_velocity_z, marker='x', color='lightgreen', label='Angular Velocity Z')
            ax2.scatter(timestamps, magnitudes, marker='*', color='lightcoral', label='Magnitude')
            ax2.set_xlabel('Timestamp')
            ax2.set_ylabel("Angular velocity",color="blue")
            ax2.legend()
            z1 = np.polyfit(timestamps, magnitudes, 1)
            p1 = np.poly1d(z1)
            ax2.plot(timestamps,p1(timestamps),color='lightcoral')
            
            i += 1
    game_index += 1       

# %% [markdown]
# ### Statistical analysis for headset rotation and motion sickness

# %%
games = ['Beat Saber','Cartoon Network', 'Monster Awakens', 'Traffic Cop', 'Voxel Shot','VR Rome','Epic Roller Coasters','Mini Motor Racing', 'Pottery']
# games = ['Beat Saber']
markers = ['o','s','^','*','x','v','+', 'd', 'p', 'h']
colors = ['r','g','b','c','m', 'k', 'aquamarine','pink','orange','purple']

game_index = 1

for game in games:   
    print(game)
    participants = []
    all_scores = []  # List to store scores for all participants in this game
    all_frames = []  # List to store frames for all participants in this game
    
    i = 0
    
    fig, axes = plt.subplots(2,3,figsize=(25,10))
    fig.suptitle(game)
    
    for filename in os.listdir(data_path):
        if not filename.startswith(".") and filename.endswith(".csv") and game in filename:

            scores = []
            frames = []
            # Get the participant id
            participant_id = filename.split(' ')[0]
            participants.append(participant_id)
            print("Participant ID: ", participant_id, " Game: ", filename.split(' ')[2], " i: ",i, " gameindex: ", game_index, " file: ", filename)    
            
            scores, frames, times = GetMotionSicknessScore(data_path + '/' + filename)
            
            angular_vel_frames, angular_velocity_x, angular_velocity_y, angular_velocity_z, magnitudes,angular_vel, timestamps = GetHeadsetAngularVelocity(data_path + '/' + filename, getAllRows=True)
            
            # get angular velocity for frames in scores
            angular_velocity_x_score = []
            angular_velocity_y_score = []
            angular_velocity_z_score = []
            for frame in frames:
                index = angular_vel_frames.index(frame)
                angular_velocity_x_score.append(angular_velocity_x[index])
                angular_velocity_y_score.append(angular_velocity_y[index])
                angular_velocity_z_score.append(angular_velocity_z[index])
            
            # calculate correlation coeffecients
            
            correlations = np.corrcoef([np.array(scores), np.array(angular_velocity_x_score), np.array(angular_velocity_y_score), np.array(angular_velocity_z_score)])
            corr_col_labels = ['Score', 'AV X', 'AV Y', 'AV Z']
            print("Correlation coefficients:")
            print(correlations)
            
            print("Plotting correlation matrix for: ", i+1)
            cax = axes[int(i/3),i%3].matshow(correlations,cmap='plasma')
            fig.colorbar(cax,ax=axes[int(i/3),i%3], label='Correlation coefficients', fraction=0.05)
            axes[int(i/3),i%3].set_title('Participant: ' + participant_id )
            axes[int(i/3),i%3].set_xticks(range(len(corr_col_labels)), corr_col_labels, rotation=45)  # Rotate x-axis labels for readability
            axes[int(i/3),i%3].set_yticks(range(len(corr_col_labels)), corr_col_labels)
               
            
            i += 1
            
            #plot scatter matrix
            # Combine data into a single DataFrame (optional)
            '''data = pd.DataFrame(correlations, columns=corr_col_labels)
            pd.plotting.scatter_matrix(data, figsize=(6, 6), alpha=0.8)
            plt.suptitle(participant_id + ' scatter matrix')'''
     
    game_index += 1            
            


# %% [markdown]
# ### Headset movement and motion sickness (Traffic Cop game)

# %%
def GetPoseData(file):
    game_name = file.split(' ')[2]
    # get participant name from csv file
    participant = file.split(' ')[1]
    print("GetPoseData: game: ", game_name, " participant: ", participant, " filename: ", file) 
    
    for item in all_dataset_files:
        if game_name in item and participant in item and 'pose.csv' in item:
            pose_data = pd.read_csv(item)
            
            # get pose data for device_id = 0
            pose_data = pose_data[pose_data['device_id'] == 0]
            
            position_cols = ['position_x', 'position_y', 'position_z']
            pose_data[position_cols] = pose_data['deviceToAbsoluteTracking'].str.split(' ', expand=True).iloc[:, [3, 7, 11]].astype(float)
            
            # extract velocity and angular velocity
            velocity_cols = ['velocity_x', 'velocity_y', 'velocity_z']
            angular_velocity_cols = ['angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z']
            
            pose_data[velocity_cols] = pose_data['velocity'].str.split(' ', expand=True).astype(float)
            pose_data[angular_velocity_cols] = pose_data['angularVelocity'].str.split(' ', expand=True).astype(float)
            '''
            print("Position x: ", pose_data['position_x'].head())
            print("Position y: ", pose_data['position_y'].head())
            print("Position z: ", pose_data['position_z'].head())
            
            print("Velocity x: ", pose_data['velocity_x'].head())
            print("Velocity y: ", pose_data['velocity_y'].head())
            print("Velocity z: ", pose_data['velocity_z'].head())
            '''
            
            # calculate linear acceleration from velocity data
            # print("diff: " , pose_data['velocity_x'].diff().head())
            pose_data['ax'] = pose_data['velocity_x'].diff() / pose_data['timestamp'].diff()
            pose_data['ay'] = pose_data['velocity_y'].diff() / pose_data['timestamp'].diff()
            pose_data['az'] = pose_data['velocity_z'].diff() / pose_data['timestamp'].diff()
            '''
            print("Linear acceleration x: ", pose_data['ax'].head())
            print("Linear acceleration y: ", pose_data['ay'].head())
            print("Linear acceleration z: ", pose_data['az'].head())
            '''
            
            # calculate jerk from acceleration
            pose_data['jerk_x'] = pose_data['ax'].diff() / pose_data['timestamp'].diff()
            pose_data['jerk_y'] = pose_data['ay'].diff() / pose_data['timestamp'].diff()
            pose_data['jerk_z'] = pose_data['az'].diff() / pose_data['timestamp'].diff()
            '''
            print("Jerk x: ", pose_data['jerk_x'].head())
            print("Jerk y: ", pose_data['jerk_y'].head())
            print("Jerk z: ", pose_data['jerk_z'].head())
            '''
            
            # calculate angular acceleration from angular velocity data
            pose_data['angular_accel_x'] = pose_data['angular_velocity_x'].diff() / pose_data['timestamp'].diff()
            pose_data['angular_accel_y'] = pose_data['angular_velocity_y'].diff() / pose_data['timestamp'].diff()
            pose_data['angular_accel_z'] = pose_data['angular_velocity_z'].diff() / pose_data['timestamp'].diff()
            '''
            print("Angular acceleration x: ", pose_data['angular_accel_x'].head())
            print("Angular acceleration y: ", pose_data['angular_accel_y'].head())
            print("Angular acceleration z: ", pose_data['angular_accel_z'].head())
            '''
            
            # calculate angular jerk from angular acceleration
            pose_data['angular_jerk_x'] = pose_data['angular_accel_x'].diff() / pose_data['timestamp'].diff()
            pose_data['angular_jerk_y'] = pose_data['angular_accel_y'].diff() / pose_data['timestamp'].diff()
            pose_data['angular_jerk_z'] = pose_data['angular_accel_z'].diff() / pose_data['timestamp'].diff()
            '''
            print("Angular jerk x: ", pose_data['angular_jerk_x'].head())
            print("Angular jerk y: ", pose_data['angular_jerk_y'].head())
            print("Angular jerk z: ", pose_data['angular_jerk_z'].head())
            '''
            
            # calculate magnitudea of angular velocity, acceleration and jerk
            pose_data['angular_velocity_magnitude'] = np.sqrt(pose_data['angular_velocity_x']**2 + pose_data['angular_velocity_y']**2 + pose_data['angular_velocity_z']**2)
            pose_data['angular_acceleration_magnitude'] = np.sqrt(pose_data['angular_accel_x']**2 + pose_data['angular_accel_y']**2 + pose_data['angular_accel_z']**2)
            pose_data['angular_jerk_magnitude'] = np.sqrt(pose_data['angular_jerk_x']**2 + pose_data['angular_jerk_y']**2 + pose_data['angular_jerk_z']**2)
            '''
            print("angular velocity magnitude: ", pose_data['angular_velocity_magnitude'].head())
            print("angular acceleration magnitude: ", pose_data['angular_acceleration_magnitude'].head())
            print("angular jerk magnitude: ", pose_data['angular_jerk_magnitude'].head())
            '''
    
    #print("Pose data: ")
    #print(pose_data.head())
    return pose_data
    

# %% [markdown]
# ### Angular velocity and motion sickness (Traffic Cop)

# %%
#games = ['Beat Saber','Cartoon Network', 'Monster Awakens', 'Traffic Cop', 'Voxel Shot','VR Rome','Epic Roller Coasters','Mini Motor Racing', 'Pottery']
games = ['Traffic Cop']
markers = ['o','s','^','*','x','v','+', 'd', 'p', 'h']
colors = ['r','g','b','c','m', 'k', 'aquamarine','pink','orange','purple']

game_index = 1

for game in games:   
    print(game)
    participants = []
    all_scores = []  # List to store scores for all participants in this game
    all_frames = []  # List to store frames for all participants in this game
    
    i = 0
    
    fig, ax = plt.subplots(2,3,figsize=(25,10))
    fig.suptitle(game)
    
    for filename in os.listdir(data_path):
        if not filename.startswith(".") and filename.endswith(".csv") and game in filename:

            scores = []
            frames = []
            # Get the participant id
            participant_id = filename.split(' ')[0]
            participants.append(participant_id)
            print("Participant ID: ", participant_id, " Game: ", filename.split(' ')[2], " i: ",i, " gameindex: ", game_index, " file: ", filename)    
            
            scores, frames, times = GetMotionSicknessScore(data_path + '/' + filename)
            
            # create pd dataframe using scores and frames
            motion_sickness_data = pd.DataFrame({'framecounter': frames, 'scores': scores})
            
            # merge motion sickness data with pose data
            pose_data = GetPoseData(data_path + '/' + filename)
            merged_data = pd.merge(motion_sickness_data, pose_data, on='framecounter')
            
            # visualize the relationship between motion sickness scores and each movement metric
            '''for col in ['scores', 'angular_velocity_magnitude', 'angular_acceleration_magnitude', 'angular_jerk_magnitude']:
                sns.scatterplot(data=merged_data, x=col, y='scores', ax=axes[int(i/3),i%3])
                axes[int(i/3),i%3].set_title(col)'''
            
            ax[int(i/3)][i%3].scatter(pose_data['framecounter'], pose_data['angular_velocity_magnitude'], marker='o', color='lightblue', label='Angular Velocity Magnitude')
            z = np.polyfit(pose_data['framecounter'], pose_data['angular_velocity_magnitude'],1)
            p = np.poly1d(z)
            ax[int(i/3)][i%3].plot(pose_data['framecounter'],p(pose_data['framecounter']),color='green')
            ax[int(i/3)][i%3].set_ylabel("Angular velocity magnitude",color="green")
            
            ax2 = ax[int(i/3)][i%3].twinx()
            ax2.scatter(frames,scores,marker=markers[i],color='blue',label='Sickness score')
            z1 = np.polyfit(frames, scores, 1)
            p1 = np.poly1d(z1)
            ax2.plot(frames,p(frames),color='blue')
            ax2.set_ylabel("Motion Sickness score",color="blue")
            '''
            ax[int(i/3)][i%3].scatter(frames,scores,marker=markers[i],color=colors[i],label='Sickness score')
            z = np.polyfit(frames, scores, 1)
            p = np.poly1d(z)
            ax[int(i/3)][i%3].plot(frames,p(frames),color=colors[i])
            ax[int(i/3)][i%3].set_ylabel("Motion Sickness score",color="green")
            
            ax2 = ax[int(i/3)][i%3].twinx()
            ax2.scatter(pose_data['framecounter'], pose_data['angular_velocity_magnitude'], marker='o', color='lightblue', label='Angular Velocity Magnitude')
            ax2.scatter(pose_data['framecounter'], pose_data['angular_acceleration_magnitude'], marker='+', color='lightgray', label='Angular Acceleration Magnitude')
            ax2.scatter(pose_data['framecounter'], pose_data['angular_jerk_magnitude'], marker='x', color='lightgreen', label='Angular Jerk Magnitude')
            '''
            i += 1
     
    game_index += 1            
            


# %% [markdown]
# ### Angular acceleration and motion sickness (Traffic Cop)

# %%
#games = ['Beat Saber','Cartoon Network', 'Monster Awakens', 'Traffic Cop', 'Voxel Shot','VR Rome','Epic Roller Coasters','Mini Motor Racing', 'Pottery']
games = ['Traffic Cop']
markers = ['o','s','^','*','x','v','+', 'd', 'p', 'h']
colors = ['r','g','b','c','m', 'k', 'aquamarine','pink','orange','purple']

game_index = 1

for game in games:   
    print(game)
    participants = []
    all_scores = []  # List to store scores for all participants in this game
    all_frames = []  # List to store frames for all participants in this game
    
    i = 0
    
    fig, ax = plt.subplots(2,3,figsize=(25,10))
    fig.suptitle(game)
    
    for filename in os.listdir(data_path):
        if not filename.startswith(".") and filename.endswith(".csv") and game in filename:

            scores = []
            frames = []
            # Get the participant id
            participant_id = filename.split(' ')[0]
            participants.append(participant_id)
            print("Participant ID: ", participant_id, " Game: ", filename.split(' ')[2], " i: ",i, " gameindex: ", game_index, " file: ", filename)    
            
            scores, frames, times = GetMotionSicknessScore(data_path + '/' + filename)
            
            # create pd dataframe using scores and frames
            motion_sickness_data = pd.DataFrame({'framecounter': frames, 'scores': scores})
            
            # merge motion sickness data with pose data
            pose_data = GetPoseData(data_path + '/' + filename)
            merged_data = pd.merge(motion_sickness_data, pose_data, on='framecounter')
            
            # visualize the relationship between motion sickness scores and each movement metric
            '''for col in ['scores', 'angular_velocity_magnitude', 'angular_acceleration_magnitude', 'angular_jerk_magnitude']:
                sns.scatterplot(data=merged_data, x=col, y='scores', ax=axes[int(i/3),i%3])
                axes[int(i/3),i%3].set_title(col)'''
            
            ax[int(i/3)][i%3].scatter(pose_data['framecounter'], pose_data['angular_acceleration_magnitude'], marker='*', color='lightgray', label='Angular Acceleration Magnitude')
            z = np.polyfit(pose_data['framecounter'], pose_data['angular_acceleration_magnitude'],1)
            p = np.poly1d(z)
            ax[int(i/3)][i%3].plot(pose_data['framecounter'],p(pose_data['framecounter']),color='green')
            ax[int(i/3)][i%3].set_ylabel("Angular acceleration magnitude",color="green")
            
            #print(pose_data['angular_acceleration_magnitude'].head())
            #ax[int(i / 3)][i % 3].set_ylim(0, 0.3)
            
            ax2 = ax[int(i/3)][i%3].twinx()
            ax2.scatter(frames,scores,marker=markers[i],color='blue',label='Sickness score')
            z1 = np.polyfit(frames, scores, 1)
            p1 = np.poly1d(z1)
            ax2.plot(frames,p1(frames),color='blue')
            ax2.set_ylabel("Motion Sickness score",color="blue")
            
            i += 1
     
    game_index += 1            
            


# %% [markdown]
# ### Angular jerk and motion sickness (Traffic Cop)

# %%
#games = ['Beat Saber','Cartoon Network', 'Monster Awakens', 'Traffic Cop', 'Voxel Shot','VR Rome','Epic Roller Coasters','Mini Motor Racing', 'Pottery']
games = ['Traffic Cop']
markers = ['o','s','^','*','x','v','+', 'd', 'p', 'h']
colors = ['r','g','b','c','m', 'k', 'aquamarine','pink','orange','purple']

game_index = 1

for game in games:   
    print(game)
    participants = []
    all_scores = []  # List to store scores for all participants in this game
    all_frames = []  # List to store frames for all participants in this game
    
    i = 0
    
    fig, ax = plt.subplots(2,3,figsize=(25,10))
    fig.suptitle(game)
    
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    
    for filename in os.listdir(data_path):
        if not filename.startswith(".") and filename.endswith(".csv") and game in filename:

            scores = []
            frames = []
            # Get the participant id
            participant_id = filename.split(' ')[0]
            participants.append(participant_id)
            print("Participant ID: ", participant_id, " Game: ", filename.split(' ')[2], " i: ",i, " gameindex: ", game_index, " file: ", filename)    
            
            scores, frames, times = GetMotionSicknessScore(data_path + '/' + filename)
            
            # create pd dataframe using scores and frames
            motion_sickness_data = pd.DataFrame({'framecounter': frames, 'scores': scores})
            
            # merge motion sickness data with pose data
            pose_data = GetPoseData(data_path + '/' + filename)
            merged_data = pd.merge(motion_sickness_data, pose_data, on='framecounter')
            
            # visualize the relationship between motion sickness scores and each movement metric
            '''for col in ['scores', 'angular_velocity_magnitude', 'angular_acceleration_magnitude', 'angular_jerk_magnitude']:
                sns.scatterplot(data=merged_data, x=col, y='scores', ax=axes[int(i/3),i%3])
                axes[int(i/3),i%3].set_title(col)'''
            
            ax[int(i/3)][i%3].scatter(pose_data['framecounter'], pose_data['angular_jerk_magnitude'], marker='x', color='lightgreen', label='Angular Jerk Magnitude')
            z = np.polyfit(pose_data['framecounter'], pose_data['angular_jerk_magnitude'],1)
            p = np.poly1d(z)
            ax[int(i/3)][i%3].plot(pose_data['framecounter'],p(pose_data['framecounter']),color='green')
            ax[int(i/3)][i%3].set_ylabel("Angular jerk magnitude",color="green")
            
            ax[int(i / 3)][i % 3].set_ylim(0, 0.0009)
            
            ax2 = ax[int(i/3)][i%3].twinx()
            ax2.scatter(frames,scores,marker=markers[i],color='blue',label='Sickness score')
            z1 = np.polyfit(frames, scores, 1)
            p1 = np.poly1d(z1)
            ax2.plot(frames,p1(frames),color='blue')
            ax2.set_ylabel("Motion Sickness score",color="blue")
         
            i += 1
     
    game_index += 1            
            


# %% [markdown]
# ### Motion sickness over time for same participants in VR Rome vs Traffic Cop

# %%
# plot graph of size 20*10 with each subgrah being 2*5
plt.figure(figsize=(20,10))

games = ['Traffic Cop','VR Rome']
markers = ['o','s']
colors = ['g','b']

gameindex = 1

# Dictionary to hold data for each participant
participant_data = {}

# for every game, plot the motion sickness scores for every participant id
# participant id is the first number in the filename
for game in games:
    for filename in os.listdir(data_path):
        if not filename.startswith(".") and filename.endswith(".csv") and game in filename:
            # Get the participant id
            participant_id = filename.split(' ')[0]
            if participant_id not in participant_data:
                participant_data[participant_id] = {}

            # Get motion sickness scores and frames
            scores, frames, _ = GetMotionSicknessScore(data_path + '/' + filename)
            participant_data[participant_id][game] = (frames, scores)

# Number of participants
num_participants = len(participant_data)
num_cols = 2  # Number of columns for subplots
num_rows = int(np.ceil(num_participants / num_cols))  # Number of rows based on participants

# Create figure and subplots
fig, axes = plt.subplots(num_rows, num_cols, figsize=(20, 10))
fig.suptitle("Motion Sickness Scores Comparison")

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.4, hspace=0.4)

# Flatten axes array for easier iteration
axes = axes.flatten()

'''
# Iterate over participants and plot their data
for idx, (participant_id, game_data) in enumerate(participant_data.items()):
    ax = axes[idx]
    for i, game in enumerate(games):
        if game in game_data:
            frames, scores = game_data[game]
            ax.scatter(frames, scores, marker=markers[i], color=colors[i], label=game)
            
            # Trend line
            z = np.polyfit(frames, scores, 1)
            p = np.poly1d(z)
            ax.plot(frames, p(frames), color=colors[i], linestyle='--', label=game + ' trendline')
    
    ax.set_title(f'Participant {participant_id}')
    ax.set_xlabel('Frame number')
    ax.set_ylabel('Motion Sickness Score')
    ax.legend()
''' 
# Iterate over participants and plot their data
for idx, (participant_id, game_data) in enumerate(participant_data.items()):
    ax = axes[idx]
    
    # Determine the full frame range for the current participant
    all_frames = np.concatenate([game_data[game][0] for game in games if game in game_data])
    frame_min, frame_max = min(all_frames), max(all_frames)
    extended_frames = np.linspace(frame_min, frame_max, num=500)  # Generate extended frame range
    
    for i, game in enumerate(games):
        if game in game_data:
            frames, scores = game_data[game]
            ax.scatter(frames, scores, marker=markers[i], color=colors[i], label=f'{game} ({participant_id})')
            
            # Trend line
            z = np.polyfit(frames, scores, 1)
            p = np.poly1d(z)
            ax.plot(extended_frames, p(extended_frames), color=colors[i], linestyle='--', label=f'{game} trendline')
    
    ax.set_title(f'Participant {participant_id}')
    ax.set_xlabel('Frame number')
    ax.set_ylabel('Motion Sickness Score')
    ax.legend()


# Remove any empty subplots
for i in range(len(participant_data), len(axes)):
    fig.delaxes(axes[i])

# Show the plot
plt.show()

# %% [markdown]
# ### Object density and motion sickness

# %%
def GetNumOfObjectsPerFrame(file):
    print("GetNumOfObjectsPerFrame: file: ", file)
    df = pd.read_csv(file)
    frames = list(set(df['framecounter']))
    name_col = df['name']
    frame_col = df['framecounter']
    total_num_obj = [] # stores an array of nums where each element corresponds to number of lights in a single frame
    objs_in_frame = []
    for i in range(len(name_col)):
        if i == 0 or frame_col[i] == frame_col[i-1]:
            # print("Added light ", name_col[i], " to frame: ", frame_col[i])
            objs_in_frame.append(name_col[i])
        else:
            total_num_obj.append(len(objs_in_frame))
            objs_in_frame = []
            objs_in_frame.append(name_col[i])
        
    total_num_obj.append(len(objs_in_frame))
    print(len(total_num_obj))
    print("frames: ", frames)
    return total_num_obj, frames


# %%
def GetObjectDensity(file):
    # get first part of game name from csv file
    game_name = file.split(' ')[2]
    # get participant name from csv file
    participant = file.split(' ')[1]
    print("GetObjectDensity: game: ", game_name, " participant: ", participant, " filename: ", file) 
    for item in all_dataset_files:
        if game_name in item and participant in item and 'object.csv' in item:
            num_of_objs_per_frame, frames = GetNumOfObjectsPerFrame(item)
            print("num of objects per frame: ", num_of_objs_per_frame)
            print("frames: ", frames)
            return num_of_objs_per_frame, frames

# %%
games = ['Beat Saber','Cartoon Network', 'Monster Awakens', 'Traffic Cop', 'Voxel Shot','VR Rome','Epic Roller Coasters','Mini Motor Racing', 'Pottery']
# games = ['Mini Motor Racing'] 
markers = ['o','s','^','*','x','v','+', 'd', 'p', 'h']
colors = ['r','g','b','c','m', 'k', 'aquamarine','pink','orange','purple']

#plt.figure(figsize=(40,20))
fig, ax = plt.subplots(9, figsize=(10,40))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
game_index = 1

for game in games:   
    print(game)
    participants = []
    all_scores = []  # List to store scores for all participants in this game
    all_frames = []  # List to store frames for all participants in this game
    all_objs = []  # List to store number of objects per frame for all participants in this game
    all_obj_frames = [] # List to store frames for all obj in this game
    
    i = 0

    for filename in os.listdir(data_path):
        if not filename.startswith(".") and filename.endswith(".csv") and game in filename:
            # if specific_participant not in report: continue
            # print(report)
            scores = []
            frames = []
            # Get the participant id
            participant_id = filename.split(' ')[0]
            participants.append(participant_id)
            print("Participant ID: ", participant_id, " Game: ", filename.split(' ')[2], " i: ",i, " gameindex: ", game_index, " file: ", filename)
            
            scores, frames, timestamps = GetMotionSicknessScore(data_path + '/' + filename)
            num_of_objs_per_frame, obj_frames = GetObjectDensity(data_path + '/' + filename)
            
            if (frames[0] != None):
                all_frames.extend(frames)
                all_scores.extend(scores)
            
            if (num_of_objs_per_frame[0] != None):
                all_objs.extend(num_of_objs_per_frame)
                
            if (obj_frames[0] != None):
                all_obj_frames.extend(obj_frames)
            
            i += 1
        else:
            continue
    
    ax[game_index-1].set_title(game)
    ax[game_index-1].scatter(all_obj_frames, all_objs, marker='+', color='lightgray')
    ax[game_index-1].set_ylabel("Number of objects per frame",color="blue",fontsize=12)
    ax[game_index-1].set_xlabel('Frame Number')
    ax2 = ax[game_index-1].twinx()
    #sns.countplot(y=num_of_objs_per_frame, ax=ax2, palette='viridis', alpha=0.7)
    ax2.scatter(all_frames,all_scores,marker=markers[(game_index-1)%(len(markers))],color=colors[(game_index-1)%(len(colors))],label=participant_id)
    ax2.set_ylabel("Motion Sickness Score",color="green",fontsize=12)
    #ax2.axis(ymin=1, ymax=5)
    ax2.set_ylim(0,6)
    
    # Plot trendlines
    if (len(all_obj_frames) > 0 and all_obj_frames[0] != None):
        z = np.polyfit(all_obj_frames, all_objs, 1)
        p = np.poly1d(z)
        ax[game_index-1].plot(all_obj_frames,p(all_obj_frames),color='g')
        #plt.plot(frames,p(frames),color='g')
                
    if (len(all_frames) > 0 and all_frames[0] != None):
        z = np.polyfit(all_frames, all_scores, 1)
        p = np.poly1d(z)
        ax2.plot(all_frames,p(all_frames),color='b')
        #plt2.plot(light_frames,p(light_frames),color='b')
    
    game_index += 1

fig.tight_layout() 



