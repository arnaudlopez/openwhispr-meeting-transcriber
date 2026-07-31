#!/bin/zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
REPO_ROOT="${SCRIPT_DIR:h}"
APP_NAME="OpenWhispr Transcribe"
BUILD_ROOT="$SCRIPT_DIR/build"
APP_BUNDLE="$BUILD_ROOT/$APP_NAME.app"
ICONSET="$BUILD_ROOT/AppIcon.iconset"
ICON_SOURCE="$SCRIPT_DIR/Assets/AppIcon.png"
INSTALL_ROOT="${HOME}/Applications"
INSTALL_BUNDLE="$INSTALL_ROOT/$APP_NAME.app"

if [[ ! -f "$ICON_SOURCE" ]]; then
  print -u2 "Missing icon source: $ICON_SOURCE"
  exit 1
fi

rm -rf -- "$APP_BUNDLE" "$ICONSET"
mkdir -p "$APP_BUNDLE/Contents/MacOS" "$APP_BUNDLE/Contents/Resources" "$ICONSET"

swiftc \
  -target arm64-apple-macosx13.0 \
  -parse-as-library \
  "$SCRIPT_DIR/Sources/OpenWhisprTranscribeApp.swift" \
  -o "$APP_BUNDLE/Contents/MacOS/$APP_NAME" \
  -framework SwiftUI \
  -framework AppKit \
  -framework AVFoundation \
  -framework Security \
  -framework UniformTypeIdentifiers

cp "$SCRIPT_DIR/Info.plist" "$APP_BUNDLE/Contents/Info.plist"
/usr/libexec/PlistBuddy \
  -c "Add :OpenWhisprProjectRoot string $REPO_ROOT" \
  "$APP_BUNDLE/Contents/Info.plist"
cp "$ICON_SOURCE" "$APP_BUNDLE/Contents/Resources/AppIcon.png"

for spec in \
  "16 icon_16x16.png" \
  "32 icon_16x16@2x.png" \
  "32 icon_32x32.png" \
  "64 icon_32x32@2x.png" \
  "128 icon_128x128.png" \
  "256 icon_128x128@2x.png" \
  "256 icon_256x256.png" \
  "512 icon_256x256@2x.png" \
  "512 icon_512x512.png" \
  "1024 icon_512x512@2x.png"
do
  size="${spec%% *}"
  filename="${spec#* }"
  sips -z "$size" "$size" "$ICON_SOURCE" --out "$ICONSET/$filename" >/dev/null
done

iconutil -c icns "$ICONSET" -o "$APP_BUNDLE/Contents/Resources/AppIcon.icns"
codesign --force --deep --sign - "$APP_BUNDLE" >/dev/null

if [[ "${1:-}" == "--install" ]]; then
  mkdir -p "$INSTALL_ROOT"
  if [[ -e "$INSTALL_BUNDLE" ]]; then
    rm -rf -- "$INSTALL_BUNDLE"
  fi
  ditto "$APP_BUNDLE" "$INSTALL_BUNDLE"
  print "$INSTALL_BUNDLE"
else
  print "$APP_BUNDLE"
fi
