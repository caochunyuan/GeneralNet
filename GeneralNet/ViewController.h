//
//  ViewController.h
//  GeneralNet
//
//  Created by Lun on 2017/3/28.
//  Copyright © 2017年 Lun. All rights reserved.
//

#import <UIKit/UIKit.h>
#import <MetalKit/MetalKit.h>
#import "GeneralNet.h"

@interface ViewController : UIViewController

@property (weak, nonatomic) IBOutlet UILabel *predictLabel;
@property (weak, nonatomic) IBOutlet UIImageView *predictView;

@property (strong, nonatomic) id <GeneralNetProtocol> alexnet;
@property (strong, nonatomic) id <GeneralNetProtocol> googlenet;
@property (strong, nonatomic) id <GeneralNetProtocol> squeezenet;

@property (assign, nonatomic) NSInteger imageNum;
@property (assign, nonatomic) NSInteger total;

- (IBAction)swipeLeft:(UISwipeGestureRecognizer *)sender;
- (IBAction)swipeRight:(UISwipeGestureRecognizer *)sender;

- (IBAction)runAlex:(id)sender;
- (IBAction)runGoogle:(id)sender;
- (IBAction)runSqueeze:(id)sender;

@end

