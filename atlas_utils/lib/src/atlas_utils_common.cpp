/**
 * Copyright 2021 Huawei Technologies Co., Ltd.
 * Copyright 2021 HiSilicon Technologies Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * You may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <memory>
#include <sys/time.h>
#include <unistd.h>
#include <iostream>
#include <fstream>
#include "acl/acl.h"
#include "acl/ops/acl_dvpp.h"
#include "atlas_utils_common.h"

using namespace std;

extern "C" {

#if 0
void* CopyDataHostToDvpp(void* data, int size) {
    void* buffer = nullptr;

    auto aclRet = acldvppMalloc(&buffer, size);
    if (aclRet != ACL_ERROR_NONE) {
        ASC_LOG_ERROR("acl malloc dvpp data failed, dataSize=%u, ret=%d", 
                      size, aclRet);
        return nullptr;
    }
    printf("malloc dvpp memory size %d ok", size);
    // copy input to device memory
    aclRet = aclrtMemcpy(buffer, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        ASC_LOG_ERROR("acl memcpy data to dvpp failed, size %u, error %d", size, aclRet);
        acldvppFree(buffer);
        return nullptr;
    }
    printf("copy data to dvpp ok");

    return buffer;
}

void* CopyDataHostToDevice(void* data, int size) {
    void* buffer = nullptr;

    auto aclRet = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (aclRet != ACL_ERROR_NONE) {
        ASC_LOG_ERROR("acl malloc device memory failed, dataSize=%u, ret=%d", 
                      size, aclRet);
        return nullptr;
    }
    // copy input to device memory
    aclRet = aclrtMemcpy(buffer, size, data, size, ACL_MEMCPY_HOST_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        ASC_LOG_ERROR("acl memcpy data to dev failed, size %u, error %d", size, aclRet);
        aclrtFree(buffer);
        return nullptr;
    }

    return buffer;
}

void* CopyDataDeviceToDevice(void* data, int size) {
    void* buffer = nullptr;

    auto aclRet = aclrtMalloc(&buffer, size, ACL_MEM_MALLOC_NORMAL_ONLY);
    if (aclRet != ACL_ERROR_NONE) {
        ASC_LOG_ERROR("acl malloc device memory failed, dataSize=%u, ret=%d", 
                      size, aclRet);
        return nullptr;
    }
    // copy input to device memory
    aclRet = aclrtMemcpy(buffer, size, data, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
    if (aclRet != ACL_ERROR_NONE) {
        ASC_LOG_ERROR("acl memcpy data to dev failed, size %u, error %d", size, aclRet);
        aclrtFree(buffer);
        return nullptr;
    }

    return buffer;
}

void* CopyDataDeviceToHost(void* deviceData, uint32_t dataLen) {  
    void *outHostData = nullptr;

    aclError ret = aclrtMallocHost(&outHostData, dataLen);
    if (ret != ACL_ERROR_NONE) {
        ASC_LOG_ERROR("aclrtMallocHost failed, ret[%d]", ret);
        return nullptr;
    }

    ret = aclrtMemcpy(outHostData, dataLen, deviceData, 
                      dataLen, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        ASC_LOG_ERROR("aclrtMemcpy failed, ret[%d]", ret);
        aclrtFreeHost(outHostData);
        return nullptr;
    }

    return outHostData;
}

void* CopyDataDeviceToNewBuf(void* deviceData, uint32_t dataLen) {  
    uint8_t* outHostData = new uint8_t[dataLen];

/*    aclError ret = aclrtMallocHost(&outHostData, dataLen);
    if (ret != ACL_ERROR_NONE) {
        ASC_LOG_ERROR("aclrtMallocHost failed, ret[%d]", ret);
        return nullptr;
    }
*/
    int ret = aclrtMemcpy(outHostData, dataLen, deviceData, 
                      dataLen, ACL_MEMCPY_DEVICE_TO_HOST);
    if (ret != ACL_ERROR_NONE) {
        ASC_LOG_ERROR("aclrtMemcpy failed, ret[%d]", ret);
        aclrtFreeHost(outHostData);
        return nullptr;
    }

    return (void *)outHostData;
}



void SaveBinFile(const char* filename, void* data, uint32_t size) {
    FILE *outFileFp = fopen(filename, "wb+");
    if (outFileFp == nullptr) {
        ASC_LOG_ERROR("Save file %s failed for open error", filename);
        return;
    }
    fwrite(data, 1, size, outFileFp);

    fflush(outFileFp);
    fclose(outFileFp);
}

char* ReadBinFile(const std::string& fileName, uint32_t& fileSize)
{
    std::ifstream binFile(fileName, std::ifstream::binary);
    if (binFile.is_open() == false) {
        ASC_LOG_ERROR("open file %s failed", fileName.c_str());
        return nullptr;
    }

    binFile.seekg(0, binFile.end);
    uint32_t binFileBufferLen = binFile.tellg();
    if (binFileBufferLen == 0) {
        ASC_LOG_ERROR("binfile is empty, filename is %s", fileName.c_str());
        binFile.close();
        return nullptr;
    }

    binFile.seekg(0, binFile.beg);

    char* binFileBufferData = new(std::nothrow) char[binFileBufferLen];
    if (binFileBufferData == nullptr) {
        ASC_LOG_ERROR("malloc binFileBufferData failed");
        binFile.close();
        return nullptr;
    }
    binFile.read(binFileBufferData, binFileBufferLen);
    binFile.close();
    fileSize = binFileBufferLen;
    return binFileBufferData;
}

int ReadImageFile(ImageData* image, const string& filename) {
    char* data;
    uint32_t size = 0;

    data = ReadBinFile(filename, size);
    if (data == nullptr) {
        ASC_LOG_ERROR("Read image file %s failed", filename.c_str());
        return STATUS_ERROR;
    }

    image->data = SHARED_PRT_U8_BUF(data);
    image->size = size;
    printf("read image ok, size %d", size);
    return STATUS_OK;
}
#endif
}
