//
//  CPUNet.h
//  GeneralNet
//
//  Created by Lun on 2017/8/19.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>
#import "GeneralNetProtocol.h"

@class CPULayer;

@interface CPUNet : NSObject <GeneralNetProtocol> {
@protected
    
    float *m_BasePtr;
    int m_Fd;
    size_t m_FileSize;
    int m_InputSize;
    unsigned char *m_ImageRawData;
    float *m_ImageData;
    float *m_ColData;
    
    CPULayer *m_FirstLayer;
    CPULayer *m_LastLayer;
    NSDictionary *m_LayersDict;
    NSArray *m_EncodeSequence;
    NSArray *m_Labels;
}

@end
