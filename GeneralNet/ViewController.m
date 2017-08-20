//
//  ViewController.m
//  GeneralNet
//
//  Created by Lun on 2017/3/28.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import "ViewController.h"

@interface ViewController ()

@end

@implementation ViewController

#pragma mark - Lifecycle Methods

- (void)viewDidLoad {
    [super viewDidLoad];
    // Do any additional setup after loading the view, typically from a nib.
    
    _imageNum = 0;
    _total = 10;
    
    // Load the appropriate Network
#if USE_METAL
    _alexnet = [MPSNet netWithDescriptionFilename:@"alexnet" dataFilename:@"metal_alexnet"];

    _googlenet = [MPSNet netWithDescriptionFilename:@"googlenet" dataFilename:@"metal_googlenet"];
    
    _squeezenet = [MPSNet netWithDescriptionFilename:@"squeezenet" dataFilename:@"metal_squeezenet"];
#else
    _alexnet = [CPUNet netWithDescriptionFilename:@"alexnet" dataFilename:@"cpu_alexnet"];
    
    _googlenet = [CPUNet netWithDescriptionFilename:@"googlenet" dataFilename:@"cpu_googlenet"];
    
    _squeezenet = [CPUNet netWithDescriptionFilename:@"squeezenet" dataFilename:@"cpu_squeezenet"];
#endif
    
    _predictView.image = [UIImage imageNamed:[[@"final" stringByAppendingString:@(_imageNum).stringValue] stringByAppendingString:@".jpg"]];
    
#if ALLOW_PRINT
    NSLog(@"Print allowed");
#else
    NSLog(@"Print forbidden");
#endif
    
#if USE_METAL
    NSLog(@"Using Metal");
#else
    NSLog(@"Using CPU");
#endif
}

- (void)didReceiveMemoryWarning {
    [super didReceiveMemoryWarning];
    // Dispose of any resources that can be recreated.
}

- (IBAction)swipeLeft:(UISwipeGestureRecognizer *)sender {
    
    // image is changing, hide predictions of previous layer
    _predictLabel.hidden = YES;
    
    // get the next image
    _imageNum = (_imageNum + 1) % _total;
    
    // display the image in UIImage View
    _predictView.image = [UIImage imageNamed:[[@"final" stringByAppendingString:@(_imageNum).stringValue] stringByAppendingString:@".jpg"]];
}

- (IBAction)swipeRight:(UISwipeGestureRecognizer *)sender {
    
    // image is changing, hide predictions of previous layer
    _predictLabel.hidden = YES;
    
    // get the previous image
    if((_imageNum - 1) >= 0) {
        _imageNum = (_imageNum - 1) % _total;
    } else{
        _imageNum = _total - 1;
    }
    
    // display the image in UIImage View
    _predictView.image = [UIImage imageNamed:[[@"final" stringByAppendingString:@(_imageNum).stringValue] stringByAppendingString:@".jpg"]];
}

- (IBAction)runAlex:(id)sender {
    [self runNet:_alexnet];
}

- (IBAction)runGoogle:(id)sender {
    [self runNet:_googlenet];
}

- (IBAction)runSqueeze:(id)sender {
    [self runNet:_squeezenet];
}

- (void)runNet:(id <GeneralNetProtocol>)net {
    NSDate *startTime = [NSDate date];
    int max_itr = 1;
    for (int i = 0; i < max_itr; i++) {
        [net forwardWithImage:self.predictView.image
                   completion:^{
                       if (i == max_itr - 1) {
                           float elapsed = -[startTime timeIntervalSinceNow] * 1000 / (float)max_itr;
                           _predictLabel.text = [[net labelsOfTopProbs] stringByAppendingFormat:@"\nElapsed time: %.0f millisecs.", elapsed];
                           _predictLabel.hidden = NO;
                       }
                   }];
    }
}

@end
