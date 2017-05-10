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
    _total = 9;
    
    // Load the appropriate Network
    _alexnet = [[GeneralNet alloc] initWithDescriptionFile:[[NSBundle mainBundle] pathForResource:@"alexnet" ofType:@"json"]
                                                  dataFile:[[NSBundle mainBundle] pathForResource:@"params_alexnet" ofType:@"dat"]];
    _googlenet = [[GeneralNet alloc] initWithDescriptionFile:[[NSBundle mainBundle] pathForResource:@"googlenet" ofType:@"json"]
                                                    dataFile:[[NSBundle mainBundle] pathForResource:@"params_googlenet" ofType:@"dat"]];
    _squeezenet = [[GeneralNet alloc] initWithDescriptionFile:[[NSBundle mainBundle] pathForResource:@"squeezenet" ofType:@"json"]
                                                     dataFile:[[NSBundle mainBundle] pathForResource:@"params_squeezenet" ofType:@"dat"]];
    
    NSString *name = [@"final" stringByAppendingString:@(_imageNum).stringValue];
    NSURL *URL = [[NSBundle mainBundle] URLForResource:name withExtension:@"jpg"];
    _predictView.image = [UIImage imageWithData:[NSData dataWithContentsOfURL:URL]];
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
    
    // get appropriate image name and path
    NSString *name = [@"final" stringByAppendingString:@(_imageNum).stringValue];
    NSURL *URL = [[NSBundle mainBundle] URLForResource:name withExtension:@"jpg"];
    // display the image in UIImage View
    _predictView.image = [UIImage imageWithData:[NSData dataWithContentsOfURL:URL]];
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
    
    // get appropriate image name and path
    NSString *name = [@"final" stringByAppendingString:@(_imageNum).stringValue];
    NSURL *URL = [[NSBundle mainBundle] URLForResource:name withExtension:@"jpg"];
    // display the image in UIImage View
    _predictView.image = [UIImage imageWithData:[NSData dataWithContentsOfURL:URL]];
}

- (IBAction)runAlex:(id)sender {
    
    NSDate *startTime = [NSDate date];
    _predictLabel.text = [[_alexnet forwardWithImage:self.predictView.image] stringByAppendingFormat:@"\nElapsed time: %.0f millisecs.", -[startTime timeIntervalSinceNow] * 1000];
    _predictLabel.hidden = NO;
}

- (IBAction)runGoogle:(id)sender {
    
    NSDate *startTime = [NSDate date];
    _predictLabel.text = [[_googlenet forwardWithImage:self.predictView.image] stringByAppendingFormat:@"\nElapsed time: %.0f millisecs.", -[startTime timeIntervalSinceNow] * 1000];
    _predictLabel.hidden = NO;
}

- (IBAction)runSqueeze:(id)sender {

    NSDate *startTime = [NSDate date];
    _predictLabel.text = [[_squeezenet forwardWithImage:self.predictView.image] stringByAppendingFormat:@"\nElapsed time: %.0f millisecs.", -[startTime timeIntervalSinceNow] * 1000];
    _predictLabel.hidden = NO;
}

@end
