import pandas as pd
import numpy as np
import time

# Modelling
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV, train_test_split
import data_prep
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
start_time_total = time.time()
start_time = time.time()
# Run the model
import Snowmeltmodel_variable

elapsed_time = time.time() - start_time

print(f"\n Elapsed time to run simulation models: {elapsed_time:.3f} seconds")
# Assign static features ---- array has length of 1x( all pixels ) x (all simulated versions)
dem = Snowmeltmodel_variable.dem_array

# Assign dynamic features ---- array has length of (timesteps) x ( all pixels ) x (all simulated versions)
snow = Snowmeltmodel_variable.snow_array
precipitation = Snowmeltmodel_variable.precipitation_array
temp = Snowmeltmodel_variable.temp_array

#Retrieve size of frame and timesteps
horizontal_pixels  = Snowmeltmodel_variable.horizontal_pixels
vertical_pixels = Snowmeltmodel_variable.vertical_pixels
timesteps = Snowmeltmodel_variable.timesteps

# Retrieve the list and the length of the list of variables used to run the different simulations
from Snowmeltmodel_variable import variable_list
list_of_variables_for_simulation = variable_list
# print(list_of_variables_for_simulation)
number_of_training_simulations = len(list_of_variables_for_simulation)
# print(number_of_training_simulations)

# Assign feature of interest to 'data'
data = snow
# Repeat Static drivers for each timestep
dem = np.repeat(dem, timesteps)

list_data = np.array_split(data, number_of_training_simulations)
list_precipitation = np.array_split(precipitation, number_of_training_simulations)
list_temp = np.array_split(temp, number_of_training_simulations)
list_dem = np.array_split(dem, number_of_training_simulations)
# create list of drivers and names
list_driver_names = ['precipitation','temp', 'dem']

dfs = [None] * number_of_training_simulations
length_one_simulation = timesteps*vertical_pixels*horizontal_pixels
# Add drivers as features
for iiii in range(number_of_training_simulations):
    print(iiii)

    dfs[iiii] = data_prep.only_y_label(data=list_data[iiii],
                                               horizontal_pixels=horizontal_pixels,
                                               vertical_pixels=vertical_pixels,
                                               multiplesteps =False)

    list_drivers = [list_precipitation[iiii],list_temp[iiii], list_dem[iiii]]

     # Add drivers at each timestep and each pixel to the dataframe
    for _ in range(len(list_drivers)):
        driver = list_drivers[_]
        name = list_driver_names[_]

        # Add driver at each timestep and each pixel to the dataframe
        dfs[iiii] = data_prep.driver_as_feature(df=dfs[iiii],
                                         driver=driver, driver_name=name,
                                         horizontal_pixels=horizontal_pixels,
                                         vertical_pixels=vertical_pixels,
                                         multiplesteps=False)

    # Add variable rate as feature
    dfs[iiii] = data_prep.VARIABLE_rate_as_feature(df= dfs[iiii],
                                                   variable_rate=list_of_variables_for_simulation[iiii],
                                                   variable_rate_name='TEMP_rate',
                                                   horizontal_pixels=horizontal_pixels,
                                                   vertical_pixels=vertical_pixels,
                                                   multiplesteps=False)

# Concatenate all dfs
df = pd.concat(dfs)

# Name features and labels
features = df.iloc[:, :-1]
labels = df.iloc[:, -1]

# Extract feature names for later processing
colnames = list(df.columns.values.tolist())
feature_names = colnames[:-1]
label_name = colnames[-1]

# Check data prep
print(feature_names)
print(features.shape)
print(label_name)
print(labels.shape)
print(number_of_training_simulations*horizontal_pixels*vertical_pixels*timesteps-(number_of_training_simulations*horizontal_pixels*vertical_pixels*2))

if labels.shape[0] == number_of_training_simulations*horizontal_pixels*vertical_pixels*timesteps-(number_of_training_simulations*horizontal_pixels*vertical_pixels*2):
    print('correct')
else:
    print('CHECK DATA PREP')

# Split train/test

length_one_simulation = timesteps*vertical_pixels*horizontal_pixels
one_time_step = horizontal_pixels*vertical_pixels

# Split at last timestep to predict last timestep
X_train = features.iloc[:-(length_one_simulation), :]
y_train = labels.iloc[:-(length_one_simulation)]

# The test set only contains the first initialised values for the simulation. The rest is predicted
X_test = features.iloc[-(length_one_simulation):(-(length_one_simulation)+one_time_step), :]

# Since we want to predict the entire simulation we do not have a set ylabel. Perhaps later construct y_label from all following simulated steps. But time series of important points generates a better view of the performance
y_test =  labels.iloc[-(length_one_simulation)+one_time_step:]

# Check train/test
print('X_train ', X_train.shape) #training features
print('Y_train ', y_train.shape) #training labels

print('X_test (only 1st step of last simulation)', X_test.shape) #testing features
print('Y_test (multiplestep Y>>X, complete simulation)', y_test.shape) #testing labels

print('Features ', features.shape) #total just to check if testing is really not inside training. shapes add up
print('Labels ', labels.shape)

optimise_model_true = False

if optimise_model_true:
    start_time_opt = time.time()

    # Altogether, there are many possible settings.
    # However, the benefit of a random search is that we are not trying every combination, but selecting at random to sample a wide range of values.

    from sklearn.model_selection import RandomizedSearchCV

    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start=10, stop=300, num=5)]
    # Number of features to consider at every split
    max_features = [1.0, 'sqrt']
    # Maximum number of levels in tree
    max_depth = [int(x) for x in np.linspace(40, 100, num=5)]

    # Minimum number of samples required to split a node
    min_samples_split = [2, 4]

    # Minimum number of samples required at each leaf node
    min_samples_leaf = [4, 8]

    # Method of selecting samples for training each tree
    bootstrap = [True]

    # Create the random grid
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}

    # Use the random grid to search for best hyperparameters
    # First create the base model to tune
    rf = RandomForestRegressor()

    # Random search of parameters, using 3 fold cross validation,
    # search across 400 different combinations, and use all available cores
    rf_random = RandomizedSearchCV(estimator=rf,
                                   param_distributions=random_grid,
                                   n_iter=4,
                                   cv=2,
                                   verbose=4,
                                   random_state=42,
                                   n_jobs=-1,
                                   scoring='explained_variance')

    # Fit the random search model
    rf_random.fit(X_train, y_train)
    print(rf_random.best_params_)
    elapsed_time = time.time() - start_time_opt
    print(f"\n Elapsed time to optimise hyperparameters of 10 sets: {elapsed_time:.3f} seconds")

else:
    # fit the model
    start_time_fitting = time.time()
    rf = RandomForestRegressor(n_estimators=15,
                               min_samples_split=4,
                               min_samples_leaf=8,
                               max_features=1.0,
                               max_depth=40,
                               bootstrap=True)
    rf.fit(X_train, y_train)
    elapsed_time = time.time() - start_time_fitting
    print(f"\n Elapsed time to fit the model: {elapsed_time:.3f} seconds")

    # Predict one complete simulation on unseen variable
    list_drivers_once = [list_precipitation[0],list_temp[0], list_dem[0]]

    # -1 timestep due to removed last timesteps to avoid NaN's in Labels
    steps = timesteps-1

    # Initialise for saving predictions for animation and performance measures, respectively.
    framed_predictions = pd.DataFrame()
    array_predictions = np.empty(0)
    list_each_prediction = [None] *steps

    #Choose the initial starting state of the model as the Test set
    X_test_multiple = X_train.iloc[:(horizontal_pixels*vertical_pixels),:]

    # - Make predictions
    # - Save prediction
    # - Set prediction as new input
    #
    # - Add static and dynamic drivers to new input
    last_pixel_driver = 0
    for _ in range(steps):
        # print(_)
        # Prediction step
        y_pred_rf = rf.predict(X_test_multiple)

        # Save the solution as a dataframe of pixels
        # For animation
        projected_prediction = pd.DataFrame(np.reshape(y_pred_rf,(int(vertical_pixels),int(horizontal_pixels))))
        framed_predictions = pd.concat([framed_predictions, projected_prediction], axis=0)

        # For MSE map
        list_each_prediction[_] = projected_prediction

        # For performance measures
        array_predictions = np.append(array_predictions, y_pred_rf)

        # Create the new test set for next prediction
        new_state = data_prep.only_y_label(y_pred_rf.reshape(-1,1),
                                                   horizontal_pixels,
                                                   vertical_pixels,
                                                   multiplesteps=True,
                                                   print_true= False)
        # Add the drivers at that timestep
        first_pixel_driver = last_pixel_driver
        last_pixel_driver = first_pixel_driver + (vertical_pixels*horizontal_pixels)
        for __ in range(len(list_drivers_once)):
            driver = list_drivers_once[__]
            name = list_driver_names[__]
            new_state = data_prep.driver_as_feature(df = new_state,
                                                    driver=driver[first_pixel_driver:(last_pixel_driver)],
                                                    driver_name= name,
                                                    horizontal_pixels=horizontal_pixels,
                                                    vertical_pixels=vertical_pixels,
                                                    multiplesteps=True)

        # Add variable rate as feature
        df = data_prep.VARIABLE_rate_as_feature(df = new_state,
                                                variable_rate= list_of_variables_for_simulation[-1],
                                                variable_rate_name = 'TEMP_rate',
                                                horizontal_pixels=horizontal_pixels,
                                                vertical_pixels=vertical_pixels,
                                                multiplesteps=True)


        # remove the Y_label-- y_label only used for training
        X_test_multiple = new_state.iloc[:, :-1]

    # Generate results:
    # MSE_map: Mean of each pixel averaged over all timesteps

    # Reshape y_test for MSE. Same shape as framed predictions. vertical x horziontal with next timestep below.
    framed_y_test = pd.DataFrame()
    first_pixel_y_test = 0
    y_test = pd.Series(y_test)
    list_y_test_ = [None] * steps

    for _ in range(steps):

        last_pixel_y_test = one_time_step*(_+1)
        # print("last pixel = ",last_pixel_y_test)
        # print("first pixel = ", first_pixel_y_test)
        # print('length y_test = ', len(y_test))
        # print('length y_test[i:j] = ', len(y_test[first_pixel_y_test:last_pixel_y_test]))
        projected_y_test = pd.DataFrame(y_test[first_pixel_y_test:last_pixel_y_test].values.reshape((int(vertical_pixels),int(horizontal_pixels))))
        list_y_test_[_] = projected_y_test
        framed_y_test = pd.concat([framed_y_test, projected_y_test], axis=0)

        first_pixel_y_test = last_pixel_y_test

    squared_difference = [None] * steps
    test_concat = pd.DataFrame()
    for __ in range(steps):
        # print(__)
        squared_difference[__] = (list_each_prediction[__] - list_y_test_[__])**2

    dfs = squared_difference
    MSE_map = pd.concat([each.stack() for each in dfs],axis=1)\
                 .apply(lambda x:x.mean(),axis=1)\
                 .unstack()


    fig, ax = plt.subplots(figsize=(5,5))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    cmap = cm.coolwarm
    im = ax.imshow(MSE_map, interpolation='nearest', cmap=cmap, vmin=MSE_map.to_numpy().max(), vmax=MSE_map.to_numpy().min())
    ax.set_title(f'MSE distribution \n  temprate = {list_of_variables_for_simulation[-1]}')
    fig.colorbar(im,cax=cax, orientation='vertical', extend = 'both')#, ticks= [0.05,0.1,0.15,0.2,0.25])

    plt.plot()
    plt.savefig(f'Results/focussed_4_training_rate{list_of_variables_for_simulation[-1]}_MSE_map.png')
    plt.show()
    plt.close()

    # overall MSE/MAPE/MAE/Explained variance
    from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, explained_variance_score,max_error

    # Mean absolute percentage error (MAPE) regression loss.
    #
    # Note here that the output is not a percentage in the range [0, 100] and a value of 100 does not mean 100% but 1e2. Furthermore, the output can be arbitrarily high when y_true is small (which is specific to the metric) or when abs(y_true - y_pred) is large (which is common for most regression metrics).


    MAPE = mean_absolute_percentage_error(y_true=y_test, y_pred = array_predictions)
    MAE = mean_absolute_error(y_true=y_test, y_pred = array_predictions)
    MSE = mean_squared_error(y_true=y_test, y_pred = array_predictions)
    MAX = max_error(y_true=y_test, y_pred = array_predictions)
    EXPL_VAR = explained_variance_score(y_true=y_test, y_pred = array_predictions)

    # print('mean absolute percentage: ', MAPE) Too small y_true values for this measure??
    print('mean absolute error: ',MAE)
    print('mean squared error: ',MSE)
    print('MAX ERROR: ',MAX)
    print('Explained variance score: ',EXPL_VAR)

    # Visualise predictions

    # Initialise the points
    point_1 = []
    point_2 = []
    point_3 = []
    point_4 = []
    point_1_pred = []
    point_2_pred = []
    point_3_pred = []
    point_4_pred = []

    # Choose the points of interest
    point_1_in_array, point_2_in_array, point_3_in_array, point_4_in_array = 139, 261, 755, 863

    # Extract the timeseries data of these points from the simulated and emulated data
    plot_xvalues = timesteps-2
    for x in range(plot_xvalues):
        point_1_in_array += int(vertical_pixels*horizontal_pixels)
        point_2_in_array += int(vertical_pixels*horizontal_pixels)
        point_3_in_array += int(vertical_pixels*horizontal_pixels)
        point_4_in_array += int(vertical_pixels*horizontal_pixels)
        point_1 = np.append(point_1, snow[point_1_in_array+((number_of_training_simulations-1)*length_one_simulation)])
        point_2 = np.append(point_2, snow[point_2_in_array+((number_of_training_simulations-1)*length_one_simulation)])
        point_3 = np.append(point_3, snow[point_3_in_array+((number_of_training_simulations-1)*length_one_simulation)])
        point_4 = np.append(point_4, snow[point_4_in_array+((number_of_training_simulations-1)*length_one_simulation)])
        point_1_pred = np.append(point_1_pred, array_predictions[point_1_in_array])
        point_2_pred = np.append(point_2_pred, array_predictions[point_2_in_array])
        point_3_pred = np.append(point_3_pred, array_predictions[point_3_in_array])
        point_4_pred = np.append(point_4_pred, array_predictions[point_4_in_array])


    fig, (ax1, ax2, ax3, ax4)= plt.subplots(4, 1, figsize=(15,15))
    fig.suptitle('Emulation performance on several points of the simulation', fontsize=18)

    #---- point 1
    ax1.plot(range(plot_xvalues), point_1, '.-', color = 'green', linewidth= 2)
    ax1.plot(range(plot_xvalues), point_1_pred, '.-', color = 'blue', linewidth= 1.0)

    #---- point 2
    ax2.plot(range(plot_xvalues), point_2, '.-', color = 'green', linewidth= 2)
    ax2.plot(range(plot_xvalues), point_2_pred, '.-', color = 'blue', linewidth= 1.0)

    #---- point 3
    ax3.plot(range(plot_xvalues), point_3, '.-', color = 'green', linewidth= 2)
    ax3.plot(range(plot_xvalues), point_3_pred, '.-', color = 'blue', linewidth= 1.0)

    #---- point 4
    ax4.plot(range(plot_xvalues), point_4, '.-', color = 'green', linewidth= 2)
    ax4.plot(range(plot_xvalues), point_4_pred, '.-', color = 'blue', linewidth= 1.0)

    ax1.grid()
    ax2.grid()
    ax3.grid()
    ax4.grid()


    ax1.set_title('First Point', fontsize = 14)
    ax1.set_ylabel('Snowfall [m]')
    ax1.set_xlabel('Timestep [day]')
    ax2.set_title('Second Point', fontsize = 14)
    ax2.set_ylabel('Snowfall [m]')
    ax2.set_xlabel('Timestep [day]')
    ax3.set_title('Third Point', fontsize = 14)
    ax3.set_ylabel('Snowfall [m]')
    ax3.set_xlabel('Timestep [day]')
    ax4.set_title('Fourth Point', fontsize = 14)
    ax4.set_ylabel('Snowfall [m]')
    ax4.set_xlabel('Timestep [day]')


    ax1.legend(['target', 'predicted'])
    ax2.legend(['target', 'predicted'])
    ax3.legend(['target', 'predicted'])
    ax4.legend(['target', 'predicted'])


    fig.tight_layout()
    # plt.subplot_tool()
    plt.plot()
    plt.savefig(f'Results/focussed4training_rate{list_of_variables_for_simulation[-1]}_timeseries.png')
    plt.show()
    plt.close()

    # # Autocorrelation plot for pattern evaluation
    # fig, (ax1, ax2, ax3, ax4)= plt.subplots(4, 1, figsize=(15,15))
    # fig.suptitle('Autocorrelation of simulation and emulation on several points', fontsize=18)
    #
    # #---- point 1
    # ax1.acorr(point_1, maxlags = plot_xvalues-1, color = 'blue', alpha = 1.0)
    # ax1.acorr(point_1_pred, maxlags = plot_xvalues-1, color = 'red', alpha=0.4, linestyle = '-')
    #
    # #---- point 2
    # ax2.acorr(point_2, maxlags = plot_xvalues-1, color = 'blue', alpha = 1.0)
    # ax2.acorr(point_2_pred, maxlags = plot_xvalues-1, color = 'red', alpha=0.4, linestyle = '-')
    #
    # #---- point 3
    # ax3.acorr(point_3, maxlags = plot_xvalues-1, color = 'blue', alpha = 1.0)
    # ax3.acorr(point_3_pred, maxlags = plot_xvalues-1, color = 'red', alpha=0.4, linestyle = '-')
    #
    # #---- point 4
    # ax4.acorr(point_4, maxlags = plot_xvalues-1, color = 'blue', alpha = 1.0)
    # ax4.acorr(point_4_pred, maxlags = plot_xvalues-1, color = 'red', alpha=0.4, linestyle = '-')
    #
    # ax1.set_title('First Point', fontsize = 14)
    # ax2.set_title('Second Point', fontsize = 14)
    # ax3.set_title('Third Point', fontsize = 14)
    # ax4.set_title('Fourth Point', fontsize = 14)
    #
    #
    # ax1.legend(['target', 'predicted'] )
    # ax2.legend(['target', 'predicted'])
    # ax3.legend(['target', 'predicted'])
    # ax4.legend(['target', 'predicted'])
    #
    #
    #
    # # fig.tight_layout()
    # # plt.subplot_tool()
    # plt.plot()
    # plt.savefig(f'Results/focussed4training_rate{list_of_variables_for_simulation[-1]}_autocorrelation.png')
    # plt.show()
    # plt.close()

    elapsed_time = time.time() - start_time_total

    print(f"Elapsed time to run whole script: {elapsed_time:.3f} seconds")
