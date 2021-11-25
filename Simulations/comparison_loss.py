''' phase_uncert_thetar simulating Optical Neural Network

using Neuroptica and linearly separable datasets
Now goes over every topology types with N = 4-32

Author: Simon Geoffroy-Gagnon
Edit: 2020.03.28
'''
import numpy as np
from sklearn.preprocessing import MinMaxScaler as mms
import calculate_accuracy as calc_acc
import ONN_Simulation_Class as ONN_Cls
import training_onn as train
import test_trained_onns as test
import ONN_Setups
import create_datasets
from copy import deepcopy
from plot_scatter_matrix import plot_scatter_matrix
import matplotlib.pyplot as plt
import neuroptica as neu

onn = ONN_Cls.ONN_Simulation()

onn.BATCH_SIZE = 400
onn.EPOCHS = 200
onn.STEP_SIZE = 0.005
onn.SAMPLES = 400

onn.ITERATIONS = 200 # number of times to retry same loss/PhaseUncert
onn.rng = 1 # starting RNG value
onn.max_number_of_tests = 25 # Max number of retries for a single model's training (keeps maximum accuracy model)
onn.max_accuracy_req = 99.9 # (%) Will stop retrying after accuracy above this is reached

onn.features = 10 # How many features? max for MNIST = 784 
onn.classes = 10 # How many classes? max for MNIST = 10
onn.N = onn.features # number of ports in device

#onn.MinMaxScaling = (0.5623, 1.7783) # For power = [-5 dB, +5 dB]
#onn.MinMaxScaling = (np.sqrt(0.1), np.sqrt(10)) # For power = [-10 dB, +10 dB]
onn.range_linear = 20
onn.range_dB = 10

#onn_topo = ['Diamond']
onn_topo = ['Diamond', 'Clements', 'Reck']
#onn_topo = ['G_H_J_F_B_C_Q_P', 'T_F_E_P', 'S_F_R_P'] #Diamond, Clements, Reck See ONN_Simulation_Class
#onn_topo = ['B_C_Q_P']

def create_model(features, classes, topo):
    ''' create ONN model based on neuroptica layer '''
    eo_settings = {'alpha': 0.1, 'g':0.5 * np.pi, 'phi_b': -1 * np.pi} # If Electro-Optic Nonlinear Activation is used

    # Some nonlinearities, to be used withing neu.Activation()
    eo_activation = neu.ElectroOpticActivation(features, **eo_settings)
    cReLU = neu.cReLU(features)
    zReLU = neu.zReLU(features)
    bpReLU = neu.bpReLU(features, cutoff=1, alpha=0.1)
    modReLU = neu.modReLU(features, cutoff=1)
    sigmoid = neu.Sigmoid(features)
    
    nlaf = cReLU # Pick the Non Linear Activation Function
    if topo == 'Diamond':
    # If you want multi-layer Diamond Topology
        model = neu.Sequential([
            # neu.AddMaskDiamond(onn.N), # Adds 0s to the top half of the Diamond input
            # neu.DiamondLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]), # Diamond Mesh
            # neu.DropMask(2*onn.N - 2, keep_ports=range(onn.N - 2, 2*onn.N - 2)), # Bottom Diamond Topology
            # neu.Activation(nlaf),
            # neu.AddMaskDiamond(onn.N), # Adds 0s to the top half of the Diamond input
            # neu.DiamondLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]), # Diamond Mesh
            # neu.DropMask(2*onn.N - 2, keep_ports=range(onn.N - 2, 2*onn.N - 2)), # Bottom Diamond Topology
            # neu.Activation(nlaf),
            # neu.AddMaskDiamond(onn.N), # Adds 0s to the top half of the Diamond input
            # neu.DiamondLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]), # Diamond Mesh
            # neu.DropMask(2*onn.N - 2, keep_ports=range(onn.N - 2, 2*onn.N - 2)), # Bottom Diamond Topology
            # neu.Activation(nlaf),
            neu.AddMaskDiamond(onn.N), # Adds 0s to the top half of the Diamond input
            neu.DiamondLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]), # Diamond Mesh
            neu.DropMask(2*onn.N - 2, keep_ports=range(onn.N - 2, 2*onn.N - 2)), # Bottom Diamond Topology
            neu.Activation(neu.AbsSquared(features)), # photodetector measurement
            #neu.DropMask(features, keep_ports=range(classes)),
        ])
    elif topo == 'Clements':
    # If you want regular Clements (multi-layer) topology
        model = neu.Sequential([
            # neu.ClementsLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            # neu.ClementsLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            # neu.ClementsLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            neu.ClementsLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            neu.Activation(neu.AbsSquared(features)), # photodetector measurement
            #neu.DropMask(features, keep_ports=range(classes))
        ])
    elif topo == 'Reck':
    # If you want regular Reck (single-layer) topology
        model = neu.Sequential([
            # neu.ReckLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            # neu.ReckLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            # neu.ReckLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            # neu.Activation(nlaf),
            neu.ReckLayer(onn.N, include_phase_shifter_layer=False, loss_dB=onn.loss_dB[0], loss_diff=onn.loss_diff, phase_uncert=onn.phase_uncert_theta[0], phases=[(None, None)]),
            neu.Activation(neu.AbsSquared(features)), # photodetector measurement
            #neu.DropMask(features, keep_ports=range(classes)) # Drops the unwanted ports
        ])
    #print(model)
    return model

accuracy_dict = []

# dataset = 'Gauss'
dataset = 'MNIST'

np.random.seed(onn.rng)
for onn.N in [10]:
    onn.features = onn.N
    onn.classes = onn.N
    loss_diff = [0.0]
    training_loss = np.linspace(0, 1, 26)


    for onn.topo in onn_topo:
        for lossDiff in loss_diff:
            if dataset == 'Gauss':
                onn, _ = train.get_dataset(onn, onn.rng, SAMPLES=400, EPOCHS=60)
            elif dataset == 'MNIST':
                onn.X, onn.y, onn.Xt, onn.yt = create_datasets.MNIST_dataset(classes=onn.classes, features=onn.features, nsamples=onn.SAMPLES)
                

            onn.FOLDER = f'Analysis/N={onn.N}'
            onn.createFOLDER()
            onn.saveSimDataset()
            temp_acc = []
            for trainLoss in training_loss:

                print(f"Training {onn.topo}")
                print("Loss Diff", lossDiff)
                print("Training Loss", trainLoss)
                max_acc = 0
                onn.loss_diff = lossDiff
                onn.loss_dB = [trainLoss]
                onn.get_topology_name()
                for test_number in range(onn.max_number_of_tests):
                    onn.phases = []

                    #model = ONN_Setups.ONN_creation(onn)
                    model = create_model(onn.features, onn.classes, onn.topo)
                    current_phases = model.get_all_phases()
                    # if onn.topo == 'Diamond':
                    #     current_phases = [[(0.9957536152885338, 3.898775749713394), (1.4695933688580787, 0.8236238034356378), (0.015323493287291465, 5.048433495956256), (0.30202906876790203, 3.2107342747530354), (0.7380871550458405, 4.293174666079528), (2.6373958327166482, 4.692096920343434), (0.7845859106719074, 3.1045264522323968), (0.06736278338608598, 2.435561137767668), (2.029437185091462, 4.153057367670428), (2.546229979666238, 5.4072007167411575), (1.3058283822110792, 1.2404251728870852), (0.0658106916905543, 1.5053688723939733), (2.022193117691199, 5.171120253802082), (1.9163921362143737, 1.9646730284115144), (0.5450882160137414, 0.14414523962451847), (0.7945821548531647, 4.315684722769842), (2.3435024836912834, 0.6344817595185769), (1.8589532993873288, 0.38464088548727343), (1.6053761655854009, 5.163726173963973), (0.7651287047180673, 5.075389861761968), (2.271780248464872, 4.92391779143677), (1.150378608017238, 3.52290524388841), (2.1152095262929964, 4.135493391757848), (1.4886637621574421, 4.5942904400604725), (1.5503916616789544, 1.1899083488439666), (1.8202229491025563, 3.4301371270201293), (2.3226748283816447, 4.951781507385526), (2.193899455251833, 2.3349777521596358), (1.3174652990533335, 0.8899806219811985), (1.0920190353632493, 0.4434367023230703), (1.488220965237406, 2.743565374576324), (1.6217741307787559, 0.29095646593359537), (2.149001090283193, 4.741321121484844), (0.4690026961363456, 1.7969895706327523), (1.1865091111128865, 0.6337330790741152), (1.747069439716735, 4.725449510144097), (0.165705645503092, 6.0239442346370495), (2.1424778496333023, 6.176382958115432), (0.018338098674082422, 3.5293561205965394), (1.400131569162883, 2.616947781204543), (0.1731432501631664, 3.0606412591569185), (0.10929129313233996, 2.906925476472582), (1.341996972180872, 0.2335432206620295), (0.14434885931505306, 4.452561506637295), (0.5550406754944661, 1.6545512933281792), (2.1520065769995176, 0.6243685231368199), (0.9010935109242181, 0.9005750053370272), (3.010992285138248, 2.4924272285292832), (0.4115868629926506, 1.5014129621347694), (0.2735000012190956, 0.5973074585720316), (2.943406409403485, 0.9783047359986121), (1.143252149863193, 4.965232667619389), (0.9344871868975759, 1.4070103936321998), (1.0877827149469366, 1.4772073276879119), (2.764856373064271, 6.1247081717989476), (1.4641641186179764, 3.3201780804719685), (1.8645714876311932, 1.423541414243275), (0.4510107541862305, 0.7656647977871042), (0.8431394852833279, 4.78410720278273), (1.992804098012697, 3.2514667497881935), (0.9373806190218031, 0.4904673181841005), (2.4324759645249814, 2.1898444821241854), (0.5842028944869398, 2.2520385308878352), (0.21857268686891201, 4.179260822307894), (2.301088251667885, 0.19481250488956503), (3.00383676090613, 1.862056772813499), (1.9067894001282677, 1.7631779983564728), (1.6550607084585438, 4.795914681540585), (1.3259307617178098, 3.1360683538565235), (0.03063633642956202, 3.9564829034580087), (0.3223008580012284, 4.329409974008857), (2.7944875810195944, 0.3671803184284466), (0.011981176449307015, 2.5180321097973133), (2.9255626444769365, 4.382466411135497), (3.0549751250699235, 0.8049242650192298), (1.1199006719416733, 4.195539549691825), (1.836135056133511, 3.302683579211228), (0.24738957786241217, 0.32440504714235385), (2.738906541455178, 3.4674188845872425), (1.0543468342044948, 0.39419267912244793), (0.41928457024014415, 4.712834391667629)], [(2.078225857487261, 5.327348666575508), (1.846311252318179, 0.14189471245581295), (2.226944959975923, 6.261647815158006), (2.2094878067530983, 3.138320863596988), (2.4165089412038565, 5.121829560347599), (0.22354158607992416, 1.6353202146551133), (1.724893153415155, 2.093450817949015), (0.44878828555725475, 1.8165560402030507), (3.0352420411481806, 0.17250966187323416), (2.2632610702365517, 5.681121021323064), (1.581434000701945, 4.117895229529219), (1.379069577241659, 2.0673934824688036), (0.4671362680849726, 1.7880882862260992), (1.6640736387315047, 1.3461224417448896), (2.6375802052180775, 5.3498697381574285), (1.5846749670351519, 5.604852156712854), (0.5310018966448566, 4.718400577266702), (2.5417226323518656, 0.35330988535539243), (2.1046072326729686, 6.067435350953189), (2.145609119338278, 5.341377408341795), (0.650759510296305, 1.19265341912275), (0.544721820052883, 4.283170691192003), (1.5348502093785616, 1.8199762998780482), (1.8145302038115658, 3.1776452242066426), (1.9907538721263727, 5.593980309271664), (1.5389624170547636, 1.8216814409254365), (2.0855280150031037, 0.44674103631134965), (2.6660365612889576, 0.10249928514657698), (0.0990964068805422, 0.9591957378313645), (2.552354958682308, 4.6811705613930705), (0.947356991505269, 2.7240225248660987), (0.8844664235284455, 1.3038844969349492), (0.8163306714941219, 3.0284192139907757), (0.5070357378835102, 5.412913939935769), (1.6059028324499072, 6.223775746715983), (1.8844649334819639, 5.251947255244975), (1.052702422918964, 3.6539968602565036), (3.0804089757793864, 3.1290793997418396), (1.8131155585346534, 4.855549645511122), (2.311215964199722, 2.427972415680304), (0.6846578721351442, 2.873658995559373), (0.10787045149104514, 5.657858056240117), (1.1929306908570751, 1.119122071606293), (1.5326967981474064, 2.415362811831494), (2.043216535097438, 5.666000324440554), (1.7512779618324172, 5.275613284617846), (2.14659450257461, 1.4621433602870115), (2.5398565870679466, 1.509358706887342), (0.529595368138562, 3.040374347971971), (0.7185148722680722, 0.517533836662591), (0.13414939960664515, 4.161968233091349), (1.3528711323844664, 0.33386400202790223), (2.7446812511211727, 4.37990295732115), (3.009328711261977, 3.2551421605579964), (1.9923500940445316, 0.6043965019181924), (1.566417313450922, 6.004691953031884), (0.6914378387645965, 4.182887531402606), (3.0792330645633177, 3.8188052827667542), (0.33047695022306217, 2.6594989873788824), (2.1852924147526402, 0.782148778562763), (3.04489967375157, 3.4310552898041706), (1.3853466033765462, 2.3868183667518594), (2.6517295371386207, 1.32204394489745), (2.511791943036099, 2.251493784213156), (1.80545627406488, 1.521265503576512), (1.981811236596088, 5.747018324415599), (1.6671378245281536, 1.6510524516412566), (2.568427564646591, 2.1828710568801224), (1.105961929507848, 4.844315414211817), (1.9456890916008758, 4.0524136975703655), (1.0200450254881663, 0.972468151646521), (0.4354916406209288, 5.164767137703222), (0.9031207232049699, 5.512890254568337), (2.9828912924890703, 4.734862648854464), (0.2949779029419753, 2.1874004238287377), (0.16148110497766024, 0.1431809836091597), (1.6429189529880726, 4.623590453964215), (0.736449960534714, 1.8417174779153316), (2.3442854273480345, 3.191771057049726), (1.283463039358832, 2.5833842531636897), (1.0610778565277128, 3.0382276184635137)]]
                    # elif onn.topo == 'Clements':
                    #     current_phases = [[(2.4076189780745416, 4.944082139745865), (0.7596908149359501, 5.6380991764852455), (2.4990866889659604, 4.972569381773501), (1.1653240339246473, 2.98104785864516), (1.838631366631236, 3.274164374462372), (2.541329604268624, 2.3394454851949082), (1.0686641675800364, 2.447124120724439), (0.7237873947342013, 2.7873268341004467), (1.5508030814773355, 5.117461171766809), (1.8452317976586763, 2.906022220365491), (0.7507243037351264, 2.1525651297303474), (1.2227548666146462, 3.8282646072471818), (0.7318952969752723, 5.1162201596984005), (0.9669583814798602, 3.0471758912741054), (2.4582304961915193, 1.5522353559765256), (0.2765994045860145, 3.004546654396125), (1.0388716435032215, 4.019582380445854), (2.012616680906371, 1.5975383079125398), (1.7021788765026205, 5.8582183763619415), (0.5350762330937067, 2.393374017201213), (2.7217111893133117, 2.626709535793292), (0.6619658848317638, 5.205933423623862), (0.8112926719394916, 0.9476288469597125), (0.24086983669873602, 2.2684256412777923), (0.5685576333756748, 5.884072296739001), (2.276468947346037, 1.4586791476996943), (0.7988987628920255, 1.7544549964186598), (1.2957785622393199, 0.03330638427232521), (2.7905783805192437, 1.1361711043213496), (2.39279332990464, 5.801408606458807), (1.9214644612294212, 0.8936509520960214), (0.09035474573046555, 2.6101642866620782), (1.876986398734764, 5.207937634975186), (2.487636066422126, 1.0961967957195606), (0.1319744328811595, 3.9875834996920982), (1.8194950002791286, 2.362716663929548), (0.658180239106944, 2.100281683480081), (1.8501025415736725, 3.225055911607511), (0.1680484425788636, 2.118234072745185), (3.086314675066232, 6.126991689260582), (0.029186488533113293, 4.1193379778217025), (0.4537195034289491, 3.6929057812121093), (0.6800301223731535, 2.4978167700268687), (2.0315924564762544, 2.1736629026843883), (1.2094285570846222, 3.0606146505522642)], [(1.4875756212590028, 2.2232204375726106), (1.8252139507192915, 0.8220536496813873), (3.0697436562932823, 2.4134116925615245), (2.164850473648287, 5.488385799825588), (3.0376495070620244, 3.468290766659634), (3.133428650192683, 2.6933000163523064), (1.305584832383189, 1.3261319861685124), (2.0478248850645766, 6.277346252289343), (1.8282185945053249, 4.033650987815978), (2.072458117054119, 5.527271573013844), (2.2881901506124827, 5.176872585456486), (0.33102975930013107, 4.367296813460738), (0.32163792486515586, 4.86620259995465), (2.540187330538196, 4.060924466331284), (1.7686785087642982, 0.08950793553602684), (2.7206130990125565, 3.5260983479592602), (2.0485913397623094, 5.05547591157946), (1.5601246621254379, 2.1945491370565393), (3.0362958114595373, 0.15797771316096265), (2.2509042333102895, 5.573037716964947), (2.448264061093972, 2.0656370140939586), (1.9444974014678924, 1.5244663697397995), (2.251928413681587, 4.348490797729798), (2.2621534676581523, 1.00582800337648), (0.6451561138965649, 4.372860934520404), (1.2955620334983697, 0.20018364529163876), (0.4042809952735608, 2.289896786652529), (0.20294181885060786, 2.7517939637080313), (2.3252541444711667, 2.1213864701826988), (0.07952584017994073, 5.682359257346997), (0.9975562295378047, 1.0136245699201791), (1.694569331586964, 2.0512305833229187), (1.7221007816818181, 2.516916360202061), (2.346811123747162, 5.964959197528747), (2.916810714102714, 3.2776629365082863), (0.39763356474441086, 1.1073227378828903), (0.08936717682766669, 0.19576292795392328), (2.066327613963378, 1.261049773909651), (2.484545854521707, 0.05914533676262473), (0.6012623770355882, 0.9333363852577314), (2.8487309992777585, 2.1330227271970914), (0.8472362034114189, 2.2047494634181835), (2.3779404385403264, 3.1490417317383126), (2.450323954157646, 0.47981441139761544), (2.401063898781144, 4.19668388213848)]]
                    # elif onn.topo == 'Reck':
                    #     current_phases = [[(2.3622155489447434, 3.5766972677939437), (2.008638197740767, 1.5149421364482978), (0.04028745889741374, 5.6649196171474845), (0.42887928521203916, 2.7473840710913016), (2.0319132303374072, 2.9235879968683944), (2.2600431797783784, 1.558263313492075), (2.02996372877043, 4.900966563915457), (1.617377713717927, 1.0270739489978524), (0.5696136587736215, 4.594177030573588), (0.7655421682797291, 1.8210021918539228), (0.8620442667510628, 0.05634523616861794), (0.20342076070474946, 1.4457162518800877), (1.7348928914843877, 5.964757377226432), (1.5673584570483863, 5.7363575461793035), (0.22907864721407606, 2.952731309004348), (2.2361901064102403, 6.086585649474679), (3.034153786523079, 0.4287697021145067), (1.25763823557186, 0.43693557750282686), (0.1024111477463018, 1.8317694211736375), (0.8442516944793996, 5.058705923069261), (2.1367794772721704, 5.2588922548271), (0.844654054448569, 4.433336512638237), (1.729947839928487, 1.5512863046010936), (0.4186813828195108, 1.577163413370666), (1.1070036939466863, 3.2839666468826305), (3.099331832987363, 3.3687553153380185), (1.889043452875912, 6.222119905503992), (1.0985190257448338, 4.068759806304794), (0.6788717609042391, 4.081304443238315), (1.8412515714504247, 3.6617480988744866), (2.879880551261044, 1.1289263577850128), (0.434993770433388, 4.221822565515999), (3.0368358947393572, 3.1082264718470336), (2.454975610253727, 3.5152503395633783), (1.2348835478186393, 6.268812573051147), (1.5046262664756178, 0.4162135715450612), (1.087595332840862, 6.1862408604088), (3.0763216801781326, 5.793477767034973), (0.2581545207158706, 5.979394095244792), (1.5686060422822243, 2.2357514554077533), (2.4612937971798985, 1.2367403679433775), (2.291252174888188, 3.728729446317438), (1.2349131299836653, 1.6043029897912717), (0.373190374291327, 4.157889221891334), (0.3489915978280976, 6.210383298169509)], [(1.580959339189132, 4.4837095538055225), (0.0921932910849526, 3.8575913023046025), (1.4899734043748922, 1.1746985278219741), (0.37059533649098303, 3.9235466586173895), (1.1878809226905205, 4.943706403169017), (0.8663084070464568, 5.066647634825221), (2.601773822946294, 0.43787160878705816), (1.3460907787361212, 3.7018355801469913), (0.012139217311791888, 2.4094068023655), (2.5314668804610765, 3.7018910835798793), (2.04092377002989, 0.7494160862827325), (2.922212526560399, 5.387800656239168), (1.7465653922154796, 4.581082712342329), (1.1610493329986948, 3.9358921550192583), (0.6476735246313611, 3.0941924597705794), (2.817278222039126, 3.2279308717353166), (2.4886662708400507, 6.1667107801425685), (2.829098379492394, 5.859608020358615), (0.2899646203574752, 2.3801122681119455), (0.7095052913035597, 6.209500568197598), (2.264184226054767, 1.2233665060702241), (2.5810056830461123, 1.6529882350018619), (3.1102699569731054, 1.0577351874893346), (2.2542481662615548, 0.6328060488461512), (0.2808489150933839, 2.5539027453646956), (1.3318249616135498, 1.1503591200451164), (0.12504092937757405, 1.845885493939288), (2.978124084103107, 6.172820134516856), (1.365924594221814, 5.954922715286129), (2.6691172875185925, 4.857099939018394), (2.449162131734657, 5.453744929583169), (0.5475086672580347, 6.063590570079559), (1.3024510979157053, 4.726607756950336), (0.9705975481573221, 3.566806274448876), (1.3589486721634265, 3.0646267093970456), (3.038920451656556, 3.7365102668657872), (0.27870270421602156, 3.683152706625404), (1.6503116519611103, 1.6224237489694329), (2.105197886388312, 2.1057375116388326), (1.6204839522442436, 6.056852731511734), (0.1799623636873749, 0.8886760392002755), (2.0079211580847063, 5.380917814146342), (0.6387580677533539, 0.9231789437500763), (2.0929372966738504, 2.8194090908760794), (1.7856667555520527, 0.9549360403126647)]]
                    # print("Starting Phases")
                    # print(current_phases)
                    model.set_all_phases_uncerts_losses(current_phases, 0, 0, trainLoss, lossDiff)
                    # print("Setting Phases")
                    # print(model.get_all_phases())
                    onn, model = train.train_single_onn(onn, model, loss_function='cce') # 'cce' for complex models, 'mse' for simple single layer ONNs
                    #print("Ending Phases")
                    #print(model.get_all_phases())
                    # Save best model
                    if max(onn.val_accuracy) > max_acc:
                        best_model = deepcopy(model)
                        best_onn = deepcopy(onn)
                        max_acc = max(onn.val_accuracy) 
                        onn.pickle_save() # save pickled version of the onn class
                        current_phases = best_model.get_all_phases()
                        best_model.set_all_phases_uncerts_losses(current_phases, 0, 0, trainLoss, lossDiff)
                    if (max(onn.val_accuracy) > onn.max_accuracy_req or
                            test_number == onn.max_number_of_tests-1):
                        print(f'\nBest Accuracy: {max_acc:.2f}%. Using this model for simulations.')
                        best_onn.loss_diff = lossDiff # Set loss_diff
                        best_onn.loss_dB = np.linspace(0, 1, 26) # set loss/MZI range
                        print(best_onn.loss_dB)
                        best_onn.phase_uncert_theta = np.linspace(0., 1, 3) # set theta phase uncert range
                        best_onn.phase_uncert_phi = np.linspace(0., 1, 3) # set phi phase uncert range

                        print('Test Accuracy of validation dataset = {:.2f}%'.format(calc_acc.accuracy(best_onn, best_model, best_onn.Xt, best_onn.yt)))

                        test.test_SLPU(best_onn, best_onn.Xt, best_onn.yt, best_model, show_progress=True)
                        temp = int(trainLoss/0.04)
                        temp_acc.append(best_onn.accuracy_LPU[temp])
                        print(temp_acc)
                        #onn.saveAll(best_model) # Save best model information
                        #onn.plotAll(trainingLoss=trainLoss) # plot training and tests
                        best_onn.plotBackprop(backprop_legend_location=0)
                        best_onn.saveForwardPropagation(best_model)
                        current_phases = best_model.get_all_phases()
                        axes = plot_scatter_matrix(best_onn.X, best_onn.y,  figsize=(15, 15), label='X', start_at=0, fontsz=54)
                        plt.savefig(best_onn.FOLDER + '/scatterplot.pdf')
                        plt.clf()
                        plt.close('all')
                        break
        accuracy_dict.append(temp_acc)
        print(accuracy_dict)

labels_size = 20
legend_size = 16
tick_size = 14
color = 'tab:blue'
fig, ax = plt.subplots(figsize=(8.27, 8.27), dpi=100) #11.69, 8.27
ax.tick_params(axis='both', which='major', labelsize=tick_size)
ax.tick_params(axis='both', which='minor', labelsize=tick_size)
ax.set_xlabel('Loss/MZI (dB)', fontsize=labels_size)
ax.set_ylabel("Validation Accuracy (\%)", fontsize=labels_size)
lns0 = ax.plot(best_onn.loss_dB, accuracy_dict[0], color='#edb120', label=onn_topo[0])
lns1 = ax.plot(best_onn.loss_dB, accuracy_dict[1], color='#d95319', label=onn_topo[1])
lns2 = ax.plot(best_onn.loss_dB, accuracy_dict[2], color='#0072bd', label=onn_topo[2])
ax.set_ylim([0, 100])
lns = lns0+lns1+lns2
labs = [l.get_label() for l in lns]
ax.legend(lns, labs, loc=0, fontsize=legend_size)
fig.tight_layout() 
plt.savefig(best_onn.FOLDER + '/comparison.pdf')
plt.clf()
plt.close('all')
