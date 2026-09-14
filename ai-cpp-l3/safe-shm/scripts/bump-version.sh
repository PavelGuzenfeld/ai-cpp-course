#!/bin/bash
# Usage: ./scripts/bump-version.sh <major|minor|patch>
# Bumps the version in VERSION, pyproject.toml, debian/changelog
set -e

VERSION_FILE="VERSION"
CURRENT=$(cat "$VERSION_FILE")
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT"

case "${1:-patch}" in
  major) MAJOR=$((MAJOR + 1)); MINOR=0; PATCH=0 ;;
  minor) MINOR=$((MINOR + 1)); PATCH=0 ;;
  patch) PATCH=$((PATCH + 1)) ;;
  *) echo "Usage: $0 <major|minor|patch>"; exit 1 ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
echo "$NEW_VERSION" > "$VERSION_FILE"

# Update pyproject.toml
if [ -f pyproject.toml ]; then
  sed -i "s/^version = \".*\"/version = \"$NEW_VERSION\"/" pyproject.toml
fi

# Update debian/changelog
if [ -f debian/changelog ]; then
  DEBFULLNAME="${DEBFULLNAME:-Pavel Guzenfeld}" \
  DEBEMAIL="${DEBEMAIL:-pavel@guzenfeld.dev}" \
  dch --newversion "${NEW_VERSION}-1" "Release ${NEW_VERSION}" 2>/dev/null || \
  sed -i "1s/([^)]*)/($NEW_VERSION-1)/" debian/changelog
fi

echo "Version bumped: $CURRENT -> $NEW_VERSION"
echo ""
echo "To release:"
echo "  git add VERSION pyproject.toml debian/changelog"
echo "  git commit -m 'Release v$NEW_VERSION'"
echo "  git tag v$NEW_VERSION"
echo "  git push origin main --tags"
