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
    
    // Load default device.
    _device = MTLCreateSystemDefaultDevice();
    
    // Make sure the current device supports MetalPerformanceShaders.
    if (!MPSSupportsMTLDevice(_device)) {
        NSLog(@"Metal Performance Shaders not Supported on current Device");
        return;
    }
    
    // Load any resources required for rendering.
    
    // Create new command queue.
    _commandQueue = [_device newCommandQueue];
    
    // make a textureLoader to get our input images as MTLTextures
    _textureLoader = [[MTKTextureLoader alloc] initWithDevice:_device];
    
    // Load the appropriate Network
    _alexnet = [[AlexNet alloc] initWithCommandQueue:_commandQueue];
    _googlenet = [[GoogleNet alloc] initWithCommandQueue:_commandQueue];
    _squeezenet = [[SqueezeNet alloc] initWithCommandQueue:_commandQueue];
    
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
    
    @autoreleasepool {
        // record time
        NSDate *startTime = [NSDate date];
        
        // load image into texture
        [self fetchImage];
        
        // encoding command buffer
        id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        
        // encode all layers of network on present commandBuffer, pass in the input image MTLTexture
        [_alexnet forwardWithCommandBuffer:commandBuffer sourceTexture:_sourceTexture];
        
        // commit the commandBuffer and wait for completion on CPU
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // display top-5 predictions for what the object should be labelled
        NSString *labels = [_alexnet getLabels];
        _predictLabel.text = [labels stringByAppendingFormat:@"\nElapsed time: %.0f millisecs.", -[startTime timeIntervalSinceNow] * 1000];
        _predictLabel.hidden = NO;
    }
}

- (IBAction)runGoogle:(id)sender {
    
    @autoreleasepool {
        // record time
        NSDate *startTime = [NSDate date];
        
        // load image into texture
        [self fetchImage];
        
        // encoding command buffer
        id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        
        // encode all layers of network on present commandBuffer, pass in the input image MTLTexture
        [_googlenet forwardWithCommandBuffer:commandBuffer sourceTexture:_sourceTexture];
        
        // commit the commandBuffer and wait for completion on CPU
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // display top-5 predictions for what the object should be labelled
        NSString *labels = [_googlenet getLabels];
        _predictLabel.text = [labels stringByAppendingFormat:@"\nElapsed time: %.0f millisecs.", -[startTime timeIntervalSinceNow] * 1000];
        _predictLabel.hidden = NO;
    }
}

- (IBAction)runSqueeze:(id)sender {
    
    @autoreleasepool {
        // record time
        NSDate *startTime = [NSDate date];
        
        // load image into texture
        [self fetchImage];
        
        // encoding command buffer
        id <MTLCommandBuffer> commandBuffer = [_commandQueue commandBuffer];
        
        // encode all layers of network on present commandBuffer, pass in the input image MTLTexture
        [_squeezenet forwardWithCommandBuffer:commandBuffer sourceTexture:_sourceTexture];
        
        // commit the commandBuffer and wait for completion on CPU
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];
        
        // display top-5 predictions for what the object should be labelled
        NSString *labels = [_squeezenet getLabels];
        _predictLabel.text = [labels stringByAppendingFormat:@"\nElapsed time: %.0f millisecs.", -[startTime timeIntervalSinceNow] * 1000];
        _predictLabel.hidden = NO;
    }
}

- (void)fetchImage {
    // get appropriate image name and path to load it into a metalTexture
    NSString *name = [@"final" stringByAppendingString:@(_imageNum).stringValue];
    NSURL *URL = [[NSBundle mainBundle] URLForResource:name withExtension:@"jpg"];
    NSError *error = NULL;
    
    _sourceTexture = [_textureLoader newTextureWithContentsOfURL:URL options:nil error:&error];
    if (error) {
        assert(error.localizedDescription);
    }
}

@end
