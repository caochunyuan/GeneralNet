//
//  GeneralNetProtocol.h
//  GeneralNet
//
//  Created by Lun on 2017/8/19.
//  Copyright © 2017年 Lun. All rights reserved.
//

#ifndef GeneralNetProtocol_h
#define GeneralNetProtocol_h

#import <Foundation/Foundation.h>
#import <UIKit/UIKit.h>

// protocol for both GPU and CPU implemention
@protocol GeneralNetProtocol

+ (id <GeneralNetProtocol>)netWithDescriptionFilename:(NSString *)descriptionFilename
                                         dataFilename:(NSString *)dataFilename;
- (void)forwardWithImage:(UIImage *)image
              completion:(void (^)())completion;
- (NSString *)labelsOfTopProbs;

@end

#endif /* GeneralNetProtocol_h */
