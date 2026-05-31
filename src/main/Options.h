#ifndef OPTIONS_H
#define OPTIONS_H

#include <string>

namespace sys {

struct Options {
  using option = unsigned char;

  struct {
    option dumpAST : 1;
    option noLink : 1;
    option dumpMidIR : 1;
    option emitIR : 1;
    option o1 : 1;
    option o2 : 1;
    option arm : 1;
    option rv : 1;
    option verbose : 1;
    option stats : 1;
    option dumpPassTiming : 1;
    option verify : 1;
    option enableExperimental : 1;
    option disableO2Experimental : 1;
    option disableLoopRotate : 1;
    option disableConstUnroll : 1;
    option enableHIRPipeline : 1;
    option useLegacyCodegen : 1;
    option forceDialectCodegen : 1;
    option dumpHIR : 1;
    option dumpCFG : 1;
    option verifyHIR : 1;
    option verifyCFG : 1;
    option bv : 1;
    option sat : 1;
    option enableRVV : 1;
    option disableSMTSynth : 1;
    option dumpAnalysisCache : 1;
    option dumpOpDescriptors : 1;
    option dumpIRContext : 1;
    option dumpPassScopes : 1;
    option dumpDialectConversion : 1;
    option dumpBlockArguments : 1;
    option dumpOperationIR : 1;
    option verifyOperationBridge : 1;
    option runSelfMLIRCoreTests : 1;
    option runSelfMLIRConversionTests : 1;
    option runSelfMLIRNativeBackendTests : 1;
    option dumpSelfMLIRSample : 1;
  };

  std::string inputFile;
  std::string outputFile;
  std::string passPipeline;
  std::string runDialectConversion;
  std::string printAfter;
  std::string printBefore;
  std::string compareWith;
  std::string simulateInput;
  std::string dialectFallbackReport;
  std::string thinSummaryOut;
  std::string thinSummaryIn;
  std::string thinLinkOut;
  std::string profileGenerate;
  std::string profileUse;
  std::string fdoUse;
  int inlineThreshold;
  int lateInlineThreshold;
  bool inlineThresholdExplicit;
  bool lateInlineThresholdExplicit;
  bool loopRotateExplicit;
  
  Options();
};

Options parseArgs(int argc, char **argv);

}

#endif
