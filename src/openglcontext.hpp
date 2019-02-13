#ifndef OPENGLCONTEXT_HPP
#define OPENGLCONTEXT_HPP

#include <EGL/egl.h>
#include <string>
#include <vector>
#include <set>
#include "gl46.h"

// Uses EGL to create a windowless OpenGL context for offscreen rendering.
class OpenGLContext {
public:
	OpenGLContext();
	~OpenGLContext();
	// Disallow copy, move, and assignment
	OpenGLContext(const OpenGLContext& o) = delete;
	OpenGLContext(OpenGLContext&& o) = delete;
	OpenGLContext& operator=(const OpenGLContext& o) = delete;
	OpenGLContext& operator=(OpenGLContext&& o) = delete;

	void makeCurrent();		// Make this context current
	bool isCurrent();		// Check if this context is current

	// Create OpenGL objects
	GLuint genVAO();
	GLuint genBuffer();
	GLuint genTexture();
	GLuint genFramebuffer();
	GLuint genRenderbuffer();
	GLuint compileShader(GLenum type, std::string source);
	GLuint linkProgram(std::vector<GLuint> shaderObjs);
	// Delete OpenGL objects
	void deleteVAO(GLuint vao);
	void deleteBuffer(GLuint buffer);
	void deleteTexture(GLuint texture);
	void deleteFramebuffer(GLuint framebuffer);
	void deleteRenderbuffer(GLuint renderbuffer);
	void deleteShader(GLuint shader);
	void deleteProgram(GLuint program);

private:
	EGLDisplay eglDpy;	// EGL context state
	EGLContext eglCtx;	// OpenGL context

	// Created OpenGL objects
	std::set<GLuint> vaos;
	std::set<GLuint> buffers;
	std::set<GLuint> textures;
	std::set<GLuint> framebuffers;
	std::set<GLuint> renderbuffers;
	std::set<GLuint> shaders;
	std::set<GLuint> programs;

	// Throws if EGL has encountered an error
	void checkEGLError(std::string what = {});
};

#endif
