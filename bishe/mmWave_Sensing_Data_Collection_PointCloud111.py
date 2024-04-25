import numpy as np
import serial
import time
import warnings
import matplotlib.pyplot as plt
from multiprocessing import Pool, Manager
from mpl_toolkits import mplot3d

# region Radar Initialization
CLIport = {}  # Initialize an empty dictionary to store the configuration parameters
Dataport = {}
byteBuffer = np.zeros(2 ** 15, dtype='uint8')
byteBufferLength = 0
configFileName = 'IWR6843_cfg_pcl.cfg'  # Configuration File
gl_numVirtAnt = 3 * 4
AoA_spacing = 1
i = 0
# endregion

# ------------------------------------------------------------------
# region Configure and Connect mmWave Radar
# Configure the serial ports
def serialConfig(configFileName):
    # Open the serial ports for the configuration and the data ports
    # CLIport = serial.Serial('/dev/COM0', 115200)
    # Dataport = serial.Serial('/dev/COM1', 921600)

    CLIport = serial.Serial('COM6', 115200)
    Dataport = serial.Serial('COM5', 921600)

    # Close the Radar for reset
    CLIport.write(('sensorStop\n').encode())
    time.sleep(0.1)
    CLIport.write(('flushCfg\n').encode())
    time.sleep(0.1)
    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        CLIport.write((i + '\n').encode())
        print(i)
        time.sleep(0.02)

    return CLIport, Dataport


# Send the data from the configuration file to our IWR6843AOP radar
def parseConfigFile(configFileName):
    configParameters = {}  # Initialize an empty dictionary to store the configuration parameters

    # Read the configuration file and send it to the board
    config = [line.rstrip('\r\n') for line in open(configFileName)]
    for i in config:
        # Split the line
        splitWords = i.split(" ")

        # Hard code the number of antennas, change if other configuration is used
        numRxAnt = 4
        numTxAnt = 3

        # Get the information about the profile configuration
        if "profileCfg" in splitWords[0]:
            startFreq = int(float(splitWords[2]))
            idleTime = int(splitWords[3])
            rampEndTime = float(splitWords[5])
            freqSlopeConst = float(splitWords[8])
            numAdcSamples = int(splitWords[10])
            numAdcSamplesRoundTo2 = 1

            while numAdcSamples > numAdcSamplesRoundTo2:
                numAdcSamplesRoundTo2 = numAdcSamplesRoundTo2 * 2

            digOutSampleRate = int(splitWords[11])

        # Get the information about the frame configuration
        elif "frameCfg" in splitWords[0]:

            chirpStartIdx = int(splitWords[1])
            chirpEndIdx = int(splitWords[2])
            numLoops = int(splitWords[3])
            numFrames = int(splitWords[4])
            framePeriodicity = float(splitWords[5])

    # Combine the read data to obtain the configuration parameters
    numChirpsPerFrame = (chirpEndIdx - chirpStartIdx + 1) * numLoops
    configParameters["numDopplerBins"] = numChirpsPerFrame // numTxAnt
    configParameters["numRangeBins"] = numAdcSamplesRoundTo2
    configParameters["rangeResolutionMeters"] = (3e8 * digOutSampleRate * 1e3) / (
            2 * freqSlopeConst * 1e12 * numAdcSamples)
    configParameters["rangeIdxToMeters"] = (3e8 * digOutSampleRate * 1e3) / (
            2 * freqSlopeConst * 1e12 * configParameters["numRangeBins"])
    configParameters["dopplerResolutionMps"] = 3e8 / (
            2 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * configParameters["numDopplerBins"] * numTxAnt)
    configParameters["maxRange"] = (300 * 0.9 * digOutSampleRate) / (2 * freqSlopeConst * 1e3)
    configParameters["maxVelocity"] = 3e8 / (4 * startFreq * 1e9 * (idleTime + rampEndTime) * 1e-6 * numTxAnt)
    configParameters["framePeriodicity"] = framePeriodicity

    rangeArray = np.array(range(configParameters["numRangeBins"])) * configParameters["rangeIdxToMeters"]
    thetaArray = np.array(range(-90, 91, AoA_spacing))
    # np.rad2deg(np.arcsin(np.array(range(-gl_numAngleBins // 2 + 1, gl_numAngleBins // 2)) *
    #                                    (2 / gl_numAngleBins)))
    dopplerArray = np.multiply(np.arange(-configParameters["numDopplerBins"] / 2,
                                         configParameters["numDopplerBins"] / 2),
                               configParameters["dopplerResolutionMps"])

    return configParameters, rangeArray, thetaArray, dopplerArray


# Read the mmWave data from the radar
def readAndParseData6843AoP(Dataport, configParameters):
    global byteBuffer, byteBufferLength

    # Constants
    OBJ_STRUCT_SIZE_BYTES = 12
    BYTE_VEC_ACC_MAX_SIZE = 2 ** 15
    MMWDEMO_UART_MSG_DETECTED_POINTS = 1
    MMWDEMO_UART_MSG_RANGE_PROFILE = 2
    MMWDEMO_OUTPUT_MSG_NOISE_PROFILE = 3
    MMWDEMO_OUTPUT_MSG_RANGE_DOPPLER_HEAT_MAP = 5
    MMWDEMO_OUTPUT_MSG_STATS = 6
    MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO = 7
    MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP = 8
    MMWDEMO_OUTPUT_MSG_TEMPERATURE_STATS = 9

    maxBufferSize = 2 ** 15
    tlvHeaderLengthInBytes = 8
    pointLengthInBytes = 16
    magicWord = [2, 1, 4, 3, 6, 5, 8, 7]

    readBuffer = Dataport.read(Dataport.in_waiting)
    byteVec = np.frombuffer(readBuffer, dtype='uint8')
    byteCount = len(byteVec)

    # Initialize variables
    magicOK = 0  # Checks if magic number has been read
    dataOK = 0  # Checks if the data has been read correctly
    dataOK_cmplx = 0
    frameNumber = 0
    numDetectedObj = 0
    detected_points = {}
    side_info = {}
    cmplx_rgAntAziEle = []

    if byteCount == 0:
        return dataOK, dataOK_cmplx, frameNumber, numDetectedObj, detected_points, side_info, cmplx_rgAntAziEle

    # Check that the buffer is not full, and then add the data to the buffer
    if (byteBufferLength + byteCount) < maxBufferSize:
        byteBuffer[byteBufferLength:byteBufferLength + byteCount] = byteVec[:byteCount]
        byteBufferLength = byteBufferLength + byteCount

    # Check that the buffer has some data
    if byteBufferLength > 16:

        # Check for all possible locations of the magic word
        possibleLocs = np.where(byteBuffer == magicWord[0])[0]

        # Confirm that is the beginning of the magic word and store the index in startIdx
        startIdx = []
        for loc in possibleLocs:
            check = byteBuffer[loc:loc + 8]
            if np.all(check == magicWord):
                startIdx.append(loc)

        # Check that startIdx is not empty
        if startIdx:

            # Remove the data before the first start index
            if 0 < startIdx[0] < byteBufferLength:
                byteBuffer[:byteBufferLength - startIdx[0]] = byteBuffer[startIdx[0]:byteBufferLength]
                byteBuffer[byteBufferLength - startIdx[0]:] = np.zeros(len(byteBuffer[byteBufferLength - startIdx[0]:]),
                                                                       dtype='uint8')
                byteBufferLength = byteBufferLength - startIdx[0]

            # Check that there have no errors with the byte buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

            # word array to convert 4 bytes to a 32 bit number
            word = [1, 2 ** 8, 2 ** 16, 2 ** 24]

            # Read the total packet length
            totalPacketLen = np.matmul(byteBuffer[12:12 + 4], word)

            # Check that all the packet has been read
            if (byteBufferLength >= totalPacketLen) and (byteBufferLength != 0):
                magicOK = 1

    # If magicOK is equal to 1 then process the message
    if magicOK:
        # Initialize the pointer index
        idX = 0

        # Read the header
        magicNumber = byteBuffer[idX:idX + 8].view(dtype=np.uint16)[0]
        idX += 8
        version = byteBuffer[idX:idX + 4].view(dtype=np.uint32)[0]
        idX += 4
        totalPacketLen = byteBuffer[idX:idX + 4].view(dtype=np.uint32)[0]
        idX += 4
        platform = byteBuffer[idX:idX + 4].view(dtype=np.uint32)[0]
        idX += 4
        frameNumber = byteBuffer[idX:idX + 4].view(dtype=np.uint32)[0]
        idX += 4
        timeCpuCycles = byteBuffer[idX:idX + 4].view(dtype=np.uint32)[0]
        idX += 4
        numDetectedObj = byteBuffer[idX:idX + 4].view(dtype=np.uint32)[0]
        idX += 4
        numTLVs = byteBuffer[idX:idX + 4].view(dtype=np.uint32)[0]
        idX += 4
        subFrameNumber = byteBuffer[idX:idX + 4].view(dtype=np.uint32)[0]
        idX += 4
        frame_header = {"magicNumber": magicNumber,
                        "version": version,
                        "totalPacketLen": totalPacketLen,
                        "platform": platform,
                        "frameNumber": frameNumber,
                        "timeCpuCycles": timeCpuCycles,
                        "numDetectedObj": numDetectedObj,
                        "numTLVs": numTLVs,
                        "subFrameNumber": subFrameNumber}

        # Read the TLV messages
        for tlvIdx in range(numTLVs):
            # print(byteBuffer[idX:idX + 4])
            # if len(byteBuffer[idX:idX + 4]) == 0:
            #     print('tlv_type is null.0000000000000000000')
            #     continue

            # Check the header of the TLV message
            tlv_type = byteBuffer[idX:idX + 4].view(dtype=np.uint32)
            idX += 4
            tlv_length = byteBuffer[idX:idX + 4].view(dtype=np.uint32)
            idX += 4
            # # word array to convert 4 bytes to a 32 bit number
            # word = [1, 2 ** 8, 2 ** 16, 2 ** 24]
            #
            # # Check the header of the TLV message
            # tlv_type = np.matmul(byteBuffer[idX:idX + 4], word)
            # idX += 4
            # tlv_length = np.matmul(byteBuffer[idX:idX + 4], word)
            # idX += 4

            # Read the data depending on the TLV message
            warnings.filterwarnings(action='ignore', category=DeprecationWarning)
            try:
                if tlv_type == MMWDEMO_UART_MSG_DETECTED_POINTS:
                    # Initialize the arrays
                    x = np.zeros(numDetectedObj, dtype=np.float32)
                    y = np.zeros(numDetectedObj, dtype=np.float32)
                    z = np.zeros(numDetectedObj, dtype=np.float32)
                    velocity = np.zeros(numDetectedObj, dtype=np.float32)

                    for objectNum in range(numDetectedObj):
                        # Read the data for each object
                        x[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)[0]
                        idX += 4
                        y[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)[0]
                        idX += 4
                        z[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)[0]
                        idX += 4
                        velocity[objectNum] = byteBuffer[idX:idX + 4].view(dtype=np.float32)[0]
                        idX += 4

                    # Store the data in the detObj dictionary
                    detected_points = {"x": x, "y": y, "z": z, "velocity": velocity}
                    dataOK += 1

                elif tlv_type == MMWDEMO_OUTPUT_MSG_DETECTED_POINTS_SIDE_INFO:
                    snr = np.zeros(numDetectedObj, dtype=np.int16)
                    noise = np.zeros(numDetectedObj, dtype=np.int16)
                    for objectNum in range(numDetectedObj):
                        snr[objectNum] = byteBuffer[idX:idX + 2].view(dtype=np.int16)[0]
                        idX += 2
                        noise[objectNum] = byteBuffer[idX:idX + 2].view(dtype=np.int16)[0]
                        idX += 2
                    side_info = {"snr": snr, "noise": noise}
                    dataOK += 1

                elif tlv_type == MMWDEMO_OUTPUT_MSG_AZIMUT_ELEVATION_STATIC_HEAT_MAP:
                    numBytes = configParameters["numRangeBins"] * gl_numVirtAnt * 4
                    range_AziEle = byteBuffer[idX:idX + numBytes].view(dtype=np.int16)

                    idX += numBytes
                    range_AziEle_img = range_AziEle[0::2].reshape((configParameters["numRangeBins"], gl_numVirtAnt))
                    range_AziEle_real = range_AziEle[1::2].reshape((configParameters["numRangeBins"], gl_numVirtAnt))
                    cmplx_rgAntAziEle = range_AziEle_real + 1j * range_AziEle_img  # configParameters["numRangeBins"] * gl_numVirtAnt

                    dataOK_cmplx += 1

                    # Some frames have strange values, skip those frames
                    # TO DO: Find why those strange frames happen
                    if np.max(np.abs(cmplx_rgAntAziEle[np.arange(1, configParameters["numRangeBins"] - 1), :])) > 10000:
                        print('############-------############')
                        print("An Outlier Frame in TI mmWave Radar.....")
                        print('############-------############')
                        # continue
                    else:
                        dataOK_cmplx += 1
            except:
                print('exception happens!')

        # Remove already processed data
        if 0 < idX < byteBufferLength:
            shiftSize = totalPacketLen

            byteBuffer[:byteBufferLength - shiftSize] = byteBuffer[shiftSize:byteBufferLength]
            byteBuffer[byteBufferLength - shiftSize:] = np.zeros(len(byteBuffer[byteBufferLength - shiftSize:]),
                                                                 dtype='uint8')
            byteBufferLength = byteBufferLength - shiftSize

            # Check that there are no errors with the buffer length
            if byteBufferLength < 0:
                byteBufferLength = 0

    return dataOK, dataOK_cmplx, frameNumber, numDetectedObj, detected_points, side_info, cmplx_rgAntAziEle


def error_handler(exception):
    print(f'{exception} occurred, terminating pool.')
    print("error: ", exception)
    pool.terminate()
# endregion


# ------------------------------------------------------------------
# region Capture Point Cloud and Azimuth-Elevation Heatmap
def Processing_GetData(mmWave_Buffer, syn_PoinyObj, ):
    # Configurate the serial ports
    CLIport, Dataport = serialConfig(configFileName)
    configParameters, rangeArray, thetaArray, dopplerArray = parseConfigFile(configFileName)

    frameNumber_next = 0
    GetData_flag = 0
    pre_pointObj = []
    while True:
        time.sleep(0.01)
        # Read and parse the received data
        dataOK, dataOK_cmplx, frameNumber, numDetectedObj, detected_points, side_info, cmplx_rgAntAziEle = \
            readAndParseData6843AoP(Dataport, configParameters)

        if frameNumber >= 1 and frameNumber > frameNumber_next:
            frameNumber_next = frameNumber

            if numDetectedObj > 0 and dataOK == 2:
                pointObj_FrameID = np.ones((numDetectedObj, 1), dtype=float) * frameNumber     # Frame ID
                pointObj_PointNum = np.ones((numDetectedObj, 1), dtype=float) * numDetectedObj # Number of Point Clouds in each frame
                pointObj_x = np.reshape(detected_points["x"], (-1, 1))   # X-axis
                pointObj_y = np.reshape(detected_points["y"], (-1, 1))   # Y-axis
                pointObj_z = np.reshape(detected_points["z"], (-1, 1))    # Z-axis
                pointObj_velocity = np.reshape(detected_points["velocity"], (-1, 1))     # Doppler Velocity
                pointObj_snr = np.reshape(side_info["snr"], (-1, 1))  # Signal-to-Noise Ratio
                pointObj = np.concatenate((pointObj_FrameID, pointObj_PointNum,
                                           pointObj_x, pointObj_y, pointObj_z,
                                           pointObj_velocity, pointObj_snr), axis=1)

                mmWave_Buffer['GetData_PointCloud'] = np.vstack((mmWave_Buffer['GetData_PointCloud'], pointObj))
                syn_PoinyObj.release()
# endregion


# ------------------------------------------------------------------
if __name__ == '__main__':
    # signal.signal(signal.SIGINT, signal.SIG_DFL)
    manager = Manager()
    syn_PoinyObj = manager.Semaphore(0)
    mmWave_Buffer = manager.dict()
    mmWave_Buffer['GetData_PointCloud'] = np.empty((0, 7), dtype=float)

    pool = Pool(processes=4)
    # Data Collection Processing
    pool.apply_async(Processing_GetData, args=(mmWave_Buffer, syn_PoinyObj,), error_callback=error_handler)

    # ----------------------------------------------------------
    while True:
        syn_PoinyObj.acquire()

        # ----------------------------------------------------------
        print('your code here')
        print(mmWave_Buffer['GetData_PointCloud'].shape)
        print(mmWave_Buffer['GetData_PointCloud'])


        # ----------------------------------------------------------

        # region Clear Buffer
        mmWave_Buffer['GetData_PointCloud'] = \
            np.delete(mmWave_Buffer['GetData_PointCloud'],
                      range(int(mmWave_Buffer['GetData_PointCloud'][0, 1])), axis=0)
        print(mmWave_Buffer['GetData_PointCloud'].shape)
        #plt
        X_pos = mmWave_Buffer['GetData_PointCloud'][:, 2]
        Y_pos = mmWave_Buffer['GetData_PointCloud'][:, 3]
        Z_pos = mmWave_Buffer['GetData_PointCloud'][:, 4]
        plt.clf()
        # plt.suptitle('Single-Target Tracking', fontsize=20)
        # plt.xlabel('X-axis (m)', fontsize=15)
        # plt.ylabel('Y-axis (m)', fontsize=15)
        # plt.xlim(-3, 3)
        # plt.ylim(0, 6)
        ax = plt.axes(projection="3d")

        ax.scatter3D(X_pos, Y_pos, Z_pos, color="red")
        plt.title("3D scatter plot")
        #plt.plot(X_pos, Y_pos, Z_pos, 'r+', markersize=1)
        # textstr_tracking = '\n'.join((
        #     r'$X= %.3f$ m' % (array_posX[-1],),
        #     r'$Y= %.3f$ m' % (array_posY[-1],),
        #     r'$V= %.3f$ m/s' % (Doppler_Speed,)))
        # plt.text(-2.8, 5.8, textstr_tracking, fontsize=12,
        #          horizontalalignment='left', verticalalignment='top', bbox=props)
        plt.pause(0.05)

        # endregion

    plt.savefig("p" + i + ".png", dpi = 300)
    i = i + 1
    plt.show()

    pool.close()
    pool.join()
