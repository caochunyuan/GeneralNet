//
//  ViewController.h
//  GeneralNet
//
//  Created by Lun on 2017/3/28.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <MetalKit/MetalKit.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import "AlexNet.h"
#import "GoogleNet.h"
#import "SqueezeNet.h"

@interface ViewController : UIViewController

// Outlets to label and view
@property (weak, nonatomic) IBOutlet UILabel *predictLabel;
@property (weak, nonatomic) IBOutlet UIImageView *predictView;

// some properties used to control the app and store appropriate values
@property (strong, nonatomic) AlexNet *alexnet;
@property (strong, nonatomic) GoogleNet *googlenet;
@property (strong, nonatomic) SqueezeNet *squeezenet;
@property (strong, nonatomic) id <MTLDevice> device;
@property (strong, nonatomic) id <MTLCommandQueue> commandQueue;
@property (strong, nonatomic) id <MTLTexture> sourceTexture;
@property (strong, nonatomic) MTKTextureLoader *textureLoader;

@property (assign, nonatomic) NSInteger imageNum;
@property (assign, nonatomic) NSInteger total;

- (IBAction)swipeLeft:(UISwipeGestureRecognizer *)sender;
- (IBAction)swipeRight:(UISwipeGestureRecognizer *)sender;
- (IBAction)runAlex:(id)sender;
- (IBAction)runGoogle:(id)sender;
- (IBAction)runSqueeze:(id)sender;

@end

