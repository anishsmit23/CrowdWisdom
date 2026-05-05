import React from "react";
import { Composition } from "remotion";
import { CWTAd, CWTAdProps } from "./CWTAd";

export const Root: React.FC = () => {
  return (
    <>
      <Composition<CWTAdProps>
        id="CWTAd"
        component={CWTAd}
        durationInFrames={1800}
        fps={30}
        width={1080}
        height={1920}
        defaultProps={{
          scenes: [],
          audioPath: "",
          subtitles: [],
        }}
      />
    </>
  );
};
