def database_of_scans(keyname):
    if keyname=='Feb':
        start_shot=70050
        end_shot=70145
        shot_list=range(start_shot,end_shot+1)
        start_time = 0.; end_time = 0.08
    elif keyname=='May5':
        start_shot=70664
        end_shot=70765
        shot_list=range(start_shot,end_shot+1)
        bad_shots=[70689,70690,70691,70698,70699,70700]
        for bad_shot_placeholder in bad_shots:
            shot_list.remove(bad_shot_placeholder)
        start_time = 0.; end_time = 0.08
    elif keyname=='May12':
        start_shot=71010
        end_shot=71058
        shot_list=range(start_shot,end_shot+1)
        start_time = 0.; end_time = 0.08
    elif keyname=='Dave':
        start_shot=58043
        end_shot=58153
        shot_list=range(start_shot,end_shot+1)
        shot_list.remove(58109)
        start_time = 0.; end_time = 0.08
    elif keyname=='May19':
        shot_list=range(71535, 71639+1)# - 1st half of the scan
        for remove_shot in range(71584,71592+1):
            shot_list.remove(remove_shot)
        shot_list.remove(71619)
        start_time = 0.; end_time = 0.08
    elif keyname=='24July2012':
        start_shot = 73758;end_shot = 73839
        shot_list=range(start_shot,end_shot+1)
        start_time = 0.; end_time = 0.08
    elif keyname=='SeptRuns':
        start_shot = 75415;end_shot = 75513
        shot_list=range(start_shot,end_shot+1)
        remove_list =range(75454, 75461+1)
        for i in remove_list:
            shot_list.remove(i)
        start_shot2 = 75707
        end_shot2 = 75789
        shot_list.extend(range(start_shot2,end_shot2+1))
        start_time = 0.; end_time = 0.09
    elif keyname=='16Aug2012FreqScan':
        #Aug16
        #frequency scaning
        #I_ring = 7424, 6940, 6490
        #I_ring = 6492, 4720, 5091, 5493, 5991
        tmp_list = [range(74845, 74865+1), range(74813, 74835+1), range(74772, 74798+1),
                    range(74757, 74763+1), range(74720, 74741+1), range(74678, 74704+1),
                    range(74644, 74671+1),range(74622, 74633+1)]
        remove_list=[74791, 74690, 74689, 74688, 74687, 74724]
        shot_list = []
        for i in tmp_list:
            for j in i:
                shot_list.append(j)
        for i in remove_list:
            shot_list.remove(i)
        start_time = 0.; end_time = 0.045
    elif keyname=='All_Shots':
        start_shot=70050; end_shot=70145
        shot_number_list1=range(start_shot,end_shot+1)
        start_shot=70664; end_shot=70765
        shot_number_list2=range(start_shot,end_shot+1)
        bad_shots=[70689,70690,70691,70698,70699,70700]
        for bad_shot_placeholder in bad_shots:
            shot_number_list2.remove(bad_shot_placeholder)
        start_shot=71010; end_shot=71058
        shot_number_list3=range(start_shot,end_shot+1)
        shot_number_list4=range(71535, 71639+1)# - 1st half of the scan
        for remove_shot in range(71584,71592+1):
            shot_number_list4.remove(remove_shot)
        shot_number_list4.remove(71619)
        shot_list = []
        for tmp_shot_list in [shot_number_list1,shot_number_list2,shot_number_list3,shot_number_list4]:
            for tmp_shot in tmp_shot_list:
                shot_list.append(tmp_shot)
        start_time = 0.; end_time = 0.08

    elif keyname=='May19Extra':
        #shot_number_list=range(71431,71448)
        #shot_number_list=range(71448,71465)
        #shot_number_list=range(71518,7153)
        #shot_number_list.remove(71516)
        #shot_number_list=range(71616,71631)
        shot_list=range(71535, 71639+1)# - 1st half of the scan
        for remove_shot in range(71584,71586+1):
            shot_list.remove(remove_shot)
        print shot_number_list
        start_time = 0.; end_time = 0.08
    elif keyname=='Test':
        shot_list=range(71728,71731)
        start_time = 0.; end_time = 0.08

    ###########################################
    ######## 5MHz scan Nov 16 2012 ###########
    elif keyname=='scan_5MHz':
        shot_list = range(76774,76807);  #16 Nov 5MHz second higher power attempt
        shot_list2 = range(76810, 76817); 
        for i in shot_list2: shot_list.append(i)
        shot_list3 = range(76843,76870); 
        for i in shot_list3: shot_list.append(i)
        remove_list = [76795, 76813, 76861]
        for i in remove_list: shot_list.remove(i)
        start_time = 0.004
        end_time = 0.040

    ###########################################
    ######## 7MHz scan Nov 15/16 2012 ###########
    elif keyname=='scan_7MHz':
        shot_list = range(76616,76662) #15 Nov 7MHz
        shot_list2 = range(76870, 76891); 
        for i in shot_list2: shot_list.append(i)
        #shot_list3 = range(76843,76870); 
        #for i in shot_list3: shot_list.append(i)
        #remove_list = [76795, 76813, 76861]
        #for i in remove_list: shot_list.remove(i)
        start_time = 0.004
        end_time = 0.040

    ###############################################
    ######## High power 7MHz scan Sept 14 2012 ####
    #This scan seems to be of dubious quality....?
    #Some interesting stuff, but lots of strange data from the instability in the transmitters
    elif keyname=='scan_7MHz_high':
        start_shot = 75415;end_shot = 75513
        shot_list=range(start_shot,end_shot+1,2)
        remove_list =range(75454, 75461+1); remove_list.append(75498)
        for i in remove_list:
            try:
                shot_list.remove(i)
            except ValueError:
                print 'removal item not in list'
        #time_bounds = [0, 0.09]
        start_time = 0.004
        end_time = 0.090

    ######################################################
    ######## High power 5MHz scan Sept 19 2012 ###########
    ####### This has good results - hints of whale tails, but mainly higher frequency activity
    elif keyname=='scan_5MHz_high':
        start_shot = 75707
        end_shot = 75789
        shot_list=range(start_shot,end_shot+1)
        start_time = 0.004
        time_bounds = [0, 0.09]

    elif keyname=='scan_4.5MHz_16Oct':
        start_shot = 80689
        #end_shot = 80791
        end_shot = 80724
        shot_list=range(start_shot,end_shot+1)
        shot_list2 = range(80761, 80795+1)
        for i in shot_list2:shot_list.append(i)
        #shot_list=range(start_shot,end_shot+1)
        remove_list = [80695]
        for i in remove_list:shot_list.remove(i)
        start_time = 0.004
        end_time = 0.079
        time_bounds = [0, 0.079]

    elif keyname=='scan_7MHz_16Oct_MP':
        start_shot = 80843
        end_shot = 80877
        shot_list=range(start_shot,end_shot+1)
        #hot_list2 = range(80761, 80795+1)
        #for i in shot_list2:shot_list.append(i)
        #shot_list=range(start_shot,end_shot+1)
        remove_list = [80847]
        for i in remove_list:shot_list.remove(i)
        start_time = 0.004
        end_time = 0.079
        time_bounds = [0, 0.079]

    ###########################################
    ######## 5MHz and 7MHz scans Nov 15/16 2012 ###########
    elif keyname=='scan_5MHz_7MHz':
        shot_list = range(76774,76807);  #16 Nov 5MHz second higher power attempt
        shot_list2 = range(76810, 76817); 
        for i in shot_list2: shot_list.append(i)
        shot_list3 = range(76843,76870); 
        for i in shot_list3: shot_list.append(i)
        remove_list = [76795, 76813, 76861]
        for i in remove_list: shot_list.remove(i)
        shot_list2 = range(76616,76662) #15 Nov 7MHz
        for i in shot_list2: shot_list.append(i)
        shot_list2 = range(76870, 76891); 
        for i in shot_list2: shot_list.append(i)
        start_time = 0.004
        end_time = 0.040

    elif keyname=='June16_2014_21inter':
        shot_list = range(83130,83212+1); 
        #shot_list2 = range(76810, 76817); 
        #for i in shot_list2: shot_list.append(i)

        remove_list = [83166,83172]
        for i in remove_list: shot_list.remove(i)
        start_time = 0.004
        end_time = 0.080
        end_time = 0.040

    elif keyname=='aug_2013_antenna_phasing':
        pass
    else:
        raise ValueError('SCAN NOT AVAILABLE!!!!!')
    return shot_list, start_time, end_time


def return_scan_details(keynames):
    '''keynames is a list of keynames for the various scans that have been done
    options include:
    scan_5MHz - Nov16 2012
    scan_7MHz - Nov15/16 2012
    scan_7MHz_high - Sept14 2012
    scan_5MHz_high - Sept19 2012
    scan_5MHz_7MHz - Nov15/16 2012
    Feb
    May5
    May12
    Dave
    May19
    24July2012
    14Sept2012
    16Aug2012FreqScan
    All_Shots
    Test
    aug_2013_antenna_phasing
    '''
    shot_list_overall = []; start_time_overall = []; end_time_overall = []
    if keynames.__class__==str:
        keynames = [keynames]
    for keyname in keynames:
        print keyname
        shot_list, start_time, end_time = database_of_scans(keyname)        
        start_time_overall.extend([start_time] * len(shot_list))
        end_time_overall.extend([end_time] * len(shot_list))
        shot_list_overall.extend(shot_list)
    return shot_list_overall, start_time_overall, end_time_overall
