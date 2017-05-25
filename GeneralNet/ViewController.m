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
    _alexnet = [[GeneralNet alloc] initWithDescriptionFile:[[NSBundle mainBundle] pathForResource:@"alexnet" ofType:@"json"]
                                                  dataFile:[[NSBundle mainBundle] pathForResource:@"metal_alexnet" ofType:@"dat"]];
    _googlenet = [[GeneralNet alloc] initWithDescriptionFile:[[NSBundle mainBundle] pathForResource:@"googlenet" ofType:@"json"]
                                                    dataFile:[[NSBundle mainBundle] pathForResource:@"metal_googlenet" ofType:@"dat"]];
    _squeezenet = [[GeneralNet alloc] initWithDescriptionFile:[[NSBundle mainBundle] pathForResource:@"squeezenet" ofType:@"json"]
                                                     dataFile:[[NSBundle mainBundle] pathForResource:@"metal_squeezenet" ofType:@"dat"]];
#else
    _alexnet = [[GeneralNet alloc] initWithDescriptionFile:[[NSBundle mainBundle] pathForResource:@"alexnet" ofType:@"json"]
                                                  dataFile:[[NSBundle mainBundle] pathForResource:@"cpu_alexnet" ofType:@"dat"]];
    _googlenet = [[GeneralNet alloc] initWithDescriptionFile:[[NSBundle mainBundle] pathForResource:@"googlenet" ofType:@"json"]
                                                    dataFile:[[NSBundle mainBundle] pathForResource:@"cpu_googlenet" ofType:@"dat"]];
    _squeezenet = [[GeneralNet alloc] initWithDescriptionFile:[[NSBundle mainBundle] pathForResource:@"squeezenet" ofType:@"json"]
                                                     dataFile:[[NSBundle mainBundle] pathForResource:@"cpu_squeezenet" ofType:@"dat"]];
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

- (void)runNet:(GeneralNet *)net {
    NSDate *startTime = [NSDate date];
    [net forwardWithImage:self.predictView.image
               completion:^{
                   float elapsed = -[startTime timeIntervalSinceNow] * 1000;
                   _predictLabel.text = [[net getTopProbs] stringByAppendingFormat:@"\nElapsed time: %.0f millisecs.", elapsed];
                   _predictLabel.hidden = NO;
               }];
}

@end
